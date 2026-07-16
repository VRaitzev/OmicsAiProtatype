# -*- coding: utf-8 -*-

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycox.models.loss import CoxPHLoss
from sksurv.metrics import concordance_index_censored
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ----------------- Reproducibility seed -----------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(max(1, os.cpu_count() or 4))
print("Seeds set:", SEED, flush=True)

N_COMPONENTS = 12
N_ITER = 2000
TOP_K = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "patients_json")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- Сети -----------------
class ServNet(nn.Module):
    """FIX: n_features передаётся явным параметром, BatchNorm применяется в forward."""
    def __init__(self, n_features=N_COMPONENTS,
                 hidden_layers_conf=[[32, 512], [512, 256], [256, 128], [128, 128], [128, 128], [128, 64]],
                 dropout=0.05):
        super().__init__()
        self.input_layer = nn.Linear(n_features, hidden_layers_conf[0][0])
        self.output_layer = nn.Linear(hidden_layers_conf[-1][1], 1)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(c[0], c[1]) for c in hidden_layers_conf])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(c[1], eps=1e-5) for c in hidden_layers_conf])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, X):
        X = self.act(self.input_layer(X))
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            X = layer(X)
            X = bn(X)          # FIX: BatchNorm теперь действительно используется
            X = self.act(X)
            X = self.dropout(X)
        return self.output_layer(X)


def init_weights_safe(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------- Данные -----------------
print("Loading data ...", flush=True)
genes_df = pd.read_csv(os.path.join(BASE_DIR, "HiSeqV2.gz"), sep="\t", index_col=0)
clin_df = pd.read_csv(os.path.join(BASE_DIR, "BRCA_survival.txt"), sep="\t", index_col=0)

# Приводим обе таблицы к виду "образцы в строках"
X_all = genes_df.T
del genes_df
X_all.index = X_all.index.astype(str)
clin_df.index = clin_df.index.astype(str)

# FIX: числовые типы для клинических полей (раньше OS.time был object -> баг при фильтрации)
clin_df = clin_df[["OS", "OS.time"]].apply(pd.to_numeric, errors="coerce")

full_df = X_all.join(clin_df, how="inner").dropna(subset=["OS", "OS.time"])
full_df = full_df[full_df["OS.time"] > 0]
full_df["OS"] = full_df["OS"].astype(int)
print("Full cohort:", full_df.shape, "events:", int(full_df['OS'].sum()), flush=True)

gene_names_all = X_all.columns.tolist()
del X_all

# ----------------- FIX: сплит ДО любой подгонки, со стратификацией по событию -----------------
idx_train, idx_test = train_test_split(
    full_df.index, test_size=0.2, random_state=SEED, stratify=full_df["OS"])
train_df = full_df.loc[idx_train]
test_df = full_df.loc[idx_test]
print(f"train={len(train_df)} (events {int(train_df['OS'].sum())}), "
      f"test={len(test_df)} (events {int(test_df['OS'].sum())})", flush=True)

# ----------------- FIX: фильтр генов по статистикам ТОЛЬКО train -----------------
mean_tr = train_df[gene_names_all].mean(axis=0)
var_tr = train_df[gene_names_all].var(axis=0)
keep = (mean_tr > 0.5) & (mean_tr < 20) & (var_tr > var_tr.quantile(0.65))
gene_names = [g for g in gene_names_all if keep[g]]
print("genes kept after filtering (train-fitted):", len(gene_names), flush=True)

X_train_raw = train_df[gene_names].values.astype(np.float64)
X_test_raw = test_df[gene_names].values.astype(np.float64)
y_time_train = train_df["OS.time"].values.astype(np.float64)
y_event_train = train_df["OS"].values.astype(np.float64)
y_time_test = test_df["OS.time"].values.astype(np.float64)
y_event_test = test_df["OS"].values.astype(np.float64)

# ----------------- NMF + стандартизация (fit только на train) -----------------
print("Fitting NMF ...", flush=True)
nmf_model = NMF(n_components=N_COMPONENTS, init="nndsvd", random_state=SEED, max_iter=N_ITER)
W_train = nmf_model.fit_transform(X_train_raw)
W_test = nmf_model.transform(X_test_raw)
H = nmf_model.components_
print("NMF done, n_iter_ =", nmf_model.n_iter_, flush=True)

components_dict = {}
for i, comp in enumerate(H):
    top_genes_idx = np.argsort(comp)[::-1][:TOP_K]
    comp_max = np.max(comp[top_genes_idx])
    components_dict[f"Component_{i + 1}"] = [
        (gene_names[j], float(comp[j] / comp_max)) for j in top_genes_idx]

scaler = StandardScaler()
X_train_np = scaler.fit_transform(W_train)
X_test_np = scaler.transform(W_test)

# ----------------- FIX: валидационная выборка из train (тест не трогаем до конца) -----------------
X_tr, X_val, t_tr, t_val, e_tr, e_val = train_test_split(
    X_train_np, y_time_train, y_event_train, test_size=0.2,
    random_state=SEED, stratify=y_event_train)

device = "cpu"
X_tr_t = torch.from_numpy(X_tr).float().to(device)
t_tr_t = torch.from_numpy(t_tr).float().to(device)
e_tr_t = torch.from_numpy(e_tr).float().to(device)
X_val_t = torch.from_numpy(X_val).float().to(device)
t_val_t = torch.from_numpy(t_val).float().to(device)
e_val_t = torch.from_numpy(e_val).float().to(device)
X_test_t = torch.from_numpy(X_test_np).float().to(device)

# ----------------- Модель -----------------
dropout = 0.05
net = ServNet(n_features=N_COMPONENTS, dropout=dropout)
net.apply(init_weights_safe)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
loss_fn = CoxPHLoss()

# ----------------- Цикл обучения (FIX: полный батч — корректные риск-сеты Кокса) -----------------
n_epochs = 100
best_val_loss = float("inf")
best_state = None
patience, bad_epochs = 20, 0

for epoch in range(1, n_epochs + 1):
    net.train()
    optimizer.zero_grad()
    out = net(X_tr_t).squeeze()
    loss = loss_fn(out, t_tr_t, e_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3.0)
    optimizer.step()

    net.eval()
    with torch.no_grad():
        out_val = net(X_val_t).squeeze()
        val_loss = loss_fn(out_val, t_val_t, e_val_t).item()
        val_cindex = concordance_index_censored(
            e_val_t.numpy().astype(bool), t_val_t.numpy(), out_val.numpy())[0]
    scheduler.step(val_loss)
    print(f"Epoch {epoch:02d} train_loss={loss.item():.4f} "
          f"val_loss={val_loss:.4f} val_cindex={val_cindex:.4f}", flush=True)

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

if best_state is not None:
    net.load_state_dict(best_state)

# ----------------- Финальная оценка на тесте -----------------
net.eval()
with torch.no_grad():
    final_out = net(X_test_t).squeeze().cpu().numpy()

y_event_bool = y_event_test.astype(bool)
cindex_direct = concordance_index_censored(y_event_bool, y_time_test, final_out)[0]
cindex_negated = concordance_index_censored(y_event_bool, y_time_test, -final_out)[0]
spearman_corr, spearman_p = spearmanr(final_out, y_time_test)
mean_event = float(final_out[y_event_bool].mean())
mean_cens = float(final_out[~y_event_bool].mean())

print("\n===== TEST RESULTS =====", flush=True)
print("C-index (linear_out, correct orientation):", cindex_direct, flush=True)
print("C-index (-linear_out, diagnostic only):", cindex_negated, flush=True)
print("Spearman(linear_out, time):", spearman_corr, "p=", spearman_p, flush=True)
print("mean linear_out (event=1):", mean_event, flush=True)
print("mean linear_out (event=0):", mean_cens, flush=True)

# ----------------- SHAP (FIX: фон — подвыборка train) -----------------
shap_values_array = None
try:
    import shap
    def model_predict(X_numpy):
        X_tensor = torch.from_numpy(np.asarray(X_numpy)).float()
        with torch.no_grad():
            return net(X_tensor).squeeze().cpu().numpy()
    background = shap.sample(X_train_np, min(100, len(X_train_np)), random_state=SEED)
    explainer = shap.Explainer(model_predict, background)
    shap_values = explainer(X_test_np)
    shap_values_array = shap_values.values
    print("SHAP computed:", shap_values_array.shape, flush=True)
except Exception as ex:
    print("SHAP failed (non-fatal):", repr(ex), flush=True)

# ----------------- Аннотации генов (защита от сетевых сбоев) -----------------
gene_desc_dict = {}
try:
    import mygene
    mg = mygene.MyGeneInfo()
    all_genes = [g for comp in components_dict.values() for g, _ in comp]
    ann = mg.querymany(all_genes, scopes="symbol", fields="name,summary", species="human")
    gene_desc_dict = {it["query"]: it.get("summary", "") for it in ann if not it.get("notfound", False)}
    print("mygene annotations:", len(gene_desc_dict), flush=True)
except Exception as ex:
    print("mygene failed (non-fatal):", repr(ex), flush=True)

# ----------------- Нормализация risk -----------------
risk_min, risk_max = final_out.min(), final_out.max()
final_out_norm = (final_out - risk_min) / (risk_max - risk_min) if risk_max > risk_min else final_out * 0.0

# ----------------- JSON для каждого пациента -----------------
for idx in range(X_test_np.shape[0]):
    patient_id = f"Patient_{idx + 1}"
    patient_dict = {"patient_id": patient_id, "risk": float(final_out_norm[idx]), "components": []}
    for comp_idx in range(N_COMPONENTS):
        comp_name = f"Component_{comp_idx + 1}"
        comp_value = float(shap_values_array[idx, comp_idx]) if shap_values_array is not None else None
        genes_list = [{"gene_name": g, "contribution": float(c),
                       "description": gene_desc_dict.get(g, "")}
                      for g, c in components_dict[comp_name]]
        patient_dict["components"].append({
            "component_id": comp_idx + 1, "value": comp_value,
            "description": f"Top genes for {comp_name}", "genes": genes_list})
    with open(os.path.join(OUT_DIR, f"{patient_id}.json"), "w") as f:
        json.dump(patient_dict, f, indent=4)

# ----------------- Сводка -----------------
summary = {
    "seed": SEED,
    "n_train": int(len(train_df)), "n_test": int(len(test_df)),
    "n_events_train": int(train_df["OS"].sum()), "n_events_test": int(test_df["OS"].sum()),
    "n_genes_after_filter": len(gene_names),
    "n_components": N_COMPONENTS,
    "cindex_test": float(cindex_direct),
    "cindex_test_negated_diagnostic": float(cindex_negated),
    "spearman_risk_time": float(spearman_corr), "spearman_p": float(spearman_p),
    "mean_risk_event": mean_event, "mean_risk_censored": mean_cens,
    "components": {k: [{"gene": g, "weight": c} for g, c in v] for k, v in components_dict.items()},
}
with open(os.path.join(BASE_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: results_summary.json and", X_test_np.shape[0], "patient JSONs in", OUT_DIR, flush=True)
print("DONE", flush=True)
