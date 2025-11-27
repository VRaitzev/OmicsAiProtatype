import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pycox.models.loss import CoxPHLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sksurv.metrics import concordance_index_censored
from scipy.stats import spearmanr
import shap
import json
import mygene

# ----------------- Reproducibility seed -----------------
patients_info = []

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED) 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Seeds set:", SEED)

mg = mygene.MyGeneInfo()
N_COMPONENETS = 12
N_ITER = 2000
# ----------------- Сети -----------------
class ServNet(nn.Module):
    def __init__(self, hidden_layers_conf=[[32,128],[128,128],[128,64]], dropout=0.1):
        super().__init__()
        self.hidden_layers_conf = hidden_layers_conf
        self.input_layer = nn.Linear(N_COMPONENETS, hidden_layers_conf[0][0])
        self.output_layer = nn.Linear(hidden_layers_conf[-1][1], 1)
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_layers_conf[i][1], eps=1e-5) 
                                          for i in range(len(hidden_layers_conf))])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        for i in range(len(hidden_layers_conf)):
            self.hidden_layers.append(nn.Linear(hidden_layers_conf[i][0], hidden_layers_conf[i][1]))

    def forward(self, X):
        X = self.input_layer(X)
        X = self.act(X)
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            X = layer(X)
            X = self.act(X)
            X = self.dropout(X)
        X = self.output_layer(X)
        return X

def init_weights_safe(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def model_predict(X_numpy):
    X_tensor = torch.from_numpy(X_numpy).float()
    with torch.no_grad():
        out = net(X_tensor).squeeze().cpu().numpy()
    return out

# ----------------- Данные -----------------
genes_df = pd.read_csv("BRCA_HiSeqV2.csv", index_col=0)
clin_df = pd.read_csv("RRCA_survival.csv", index_col=0)
common = genes_df.columns.intersection(clin_df.columns)
genes_df = genes_df[common]
clin_df = clin_df[common]

genes_df = genes_df.apply(pd.to_numeric, errors="coerce")
genes_df["mean_expression"] = genes_df.mean(axis=1)
genes_df["variance"] = genes_df.var(axis=1)
genes_df = genes_df[(genes_df['mean_expression']>0.5) & (genes_df['mean_expression']<20)]
genes_df = genes_df[genes_df['variance']>genes_df['variance'].quantile(0.65)]
genes_df = genes_df.drop(columns=["mean_expression","variance"])

genes_df = genes_df.T
clin_df = clin_df.T
genes_df.index = genes_df.index.astype(str)
clin_df.index = clin_df.index.astype(str)
genes_df = genes_df.sort_index()
clin_df = clin_df.sort_index()

full_df = genes_df.join(clin_df, how="inner").dropna(subset=["OS.time"])
full_df = full_df[full_df["OS.time"]>0]

X = full_df.drop(columns=clin_df.columns).values
y_event = torch.from_numpy(full_df["OS"].values).float()
y_time = torch.from_numpy(full_df["OS.time"].values).float()

# ----------------- NMF + стандартизация -----------------
nmf_model = NMF(n_components=N_COMPONENETS, init='nndsvd', random_state=SEED, max_iter=N_ITER)
W = nmf_model.fit_transform(X)
H = nmf_model.components_

gene_names = full_df.drop(columns=clin_df.columns).columns.tolist()

top_k = 20  
components_dict = {}

for i, comp in enumerate(H):
    top_genes_idx = np.argsort(comp)[::-1][:top_k]
    comp_max = np.max(comp[top_genes_idx])
    top_genes_with_values = [(gene_names[j], float(comp[j]/comp_max)) for j in top_genes_idx]
    components_dict[f"Component_{i+1}"] = top_genes_with_values


scaler = StandardScaler()
X_deep = scaler.fit_transform(W)

X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X_deep, y_time.numpy(), y_event.numpy(), test_size=0.2, random_state=SEED
)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = (torch.from_numpy(y_time_train).float(), torch.from_numpy(y_event_train).float())
y_test = (torch.from_numpy(y_time_test).float(), torch.from_numpy(y_event_test).float())

# ----------------- Модель -----------------
hidden_layers_conf = [    
    [32, 512],
    [512, 256],
    [256, 128],
    [128, 128],
    [128, 128],
    [128, 64]
    ]
dropout = 0.05
net = ServNet( hidden_layers_conf=hidden_layers_conf, dropout=dropout)
net.apply(init_weights_safe)
device = 'cpu'
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
loss_fn = CoxPHLoss()

train_dataset = TensorDataset(X_train, y_train[0], y_train[1])
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g, num_workers=0)

X_test_tensor = X_test.to(device)
t_test = y_test[0].to(device)
e_test = y_test[1].to(device)

# ----------------- Цикл обучения -----------------
n_epochs = 100
best_val_loss = float('inf')
best_state = None

for epoch in range(1, n_epochs+1):
    net.train()
    epoch_loss = 0.0
    n_batches = 0
    for Xb, tb, eb in train_loader:
        Xb, tb, eb = Xb.to(device), tb.to(device), eb.to(device)
        optimizer.zero_grad()
        out = net(Xb).squeeze()
        loss = loss_fn(out, tb, eb)            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3.0)  
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    epoch_loss /= max(1,n_batches)

    net.eval()
    with torch.no_grad():
        out_test = net(X_test_tensor).squeeze()
        val_loss = loss_fn(out_test, t_test, e_test).item()   
        preds_np = out_test.cpu().numpy()
        y_time_np = t_test.cpu().numpy()
        y_event_np = e_test.cpu().numpy().astype(bool)
        val_cindex = concordance_index_censored(y_event_np, y_time_np, preds_np)[0]

    scheduler.step(val_loss)
    print(f"Epoch {epoch:02d} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_cindex={val_cindex:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k,v in net.state_dict().items()}

if best_state is not None:
    net.load_state_dict(best_state)

# ----------------- Финальная оценка -----------------
net.eval()
with torch.no_grad():
    final_out = net(X_test_tensor).squeeze().cpu().numpy()

print("Final linear_out shape:", final_out.shape)
y_time_np = t_test.cpu().numpy()
y_event_np = e_test.cpu().numpy().astype(bool)
print("Final C-index (linear_out):", concordance_index_censored(y_event_np, y_time_np, final_out)[0])
print("Final C-index (-linear_out):", concordance_index_censored(y_event_np, y_time_np, -final_out)[0])
spearman_corr, spearman_p = spearmanr(final_out, y_time_np)
print("Spearman(linear_out, time):", spearman_corr, "p=", spearman_p)
mean_event = final_out[y_event_np==1].mean()
mean_cens  = final_out[y_event_np==0].mean()
print("mean linear_out (event=1):", mean_event)
print("mean linear_out (event=0):", mean_cens)

explainer = shap.Explainer(model_predict, X_train.numpy())

shap_values = explainer(X_test.numpy())

shap_values_array = shap_values.values 
max_abs = np.max(np.abs(shap_values_array))
shap_normalized = shap_values_array / max_abs

# ----------------- Подготовка аннотаций для всех генов -----------------
all_genes = [gene_name for comp in components_dict.values() for gene_name, _ in comp]
gene_annotations = mg.querymany(all_genes, scopes="symbol", fields="name,summary", species="human")

gene_desc_dict = {item['query']: item.get('summary', "") for item in gene_annotations if not item.get('notfound', False)}

# ----------------- Нормализация risk -----------------
risk_min = final_out.min()
risk_max = final_out.max()
final_out_norm = (final_out - risk_min) / (risk_max - risk_min)

# ----------------- Создание JSON для каждого пациента -----------------
shap_values_array = shap_values.values  

for idx in range(X_test.shape[0]):
    patient_id = f"Patient_{idx+1}"
    patient_risk = float(final_out_norm[idx]) 

    patient_dict = {
        "patient_id": patient_id,
        "risk": patient_risk,
        "components": []
    }

    for comp_idx in range(N_COMPONENETS):
        comp_name = f"Component_{comp_idx+1}"
        comp_value = float(shap_values_array[idx, comp_idx]) 

        genes_list = []
        for gene_name, contribution in components_dict[comp_name]:
            genes_list.append({
                "gene_name": gene_name,
                "contribution": float(contribution),
                "description": gene_desc_dict.get(gene_name, "")
            })

        patient_dict["components"].append({
            "component_id": comp_idx+1,
            "value": comp_value,
            "description": f"Top genes for {comp_name}",
            "genes": genes_list
        })

    with open(f"{patient_id}.json", "w") as f:
        json.dump(patient_dict, f, indent=4)
