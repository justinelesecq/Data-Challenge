import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from Classification_MLP_architecture import MLP_CE, train_loop_CE, running_classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#"""""""""""""""""""""""""""""""""""""" CHARGEMENTS DES DONNÉES D'ENTREE ET DE TEST """"""""""""""""""""""""

data_train = pd.read_csv("data_train_clean.csv")
data_test = pd.read_csv("data_test_clean.csv")

data_train = data_train.drop(columns=["Unnamed: 0"])
data_test = data_test.drop(columns=["Unnamed: 0"])

#"""""""""""""""""""""""""""""""""""""""""""""""""""""" ADD FEATURES """"""""""""""""""""""""""""""""""""""""""""""""

def add_features(df):
    df = df.copy()
    inc = df["income"].where(df["income"] > 0) * 1000      # income est en milliers de $
    
    df["loan_to_income"]   = df["loan_amount"] / inc
    df["income_vs_msa"]    = inc / df["ffiec_msa_md_median_family_income"]
    df["loan_to_value_calc"] = df["loan_amount"] / df["property_value"]
    df["implied_value"]    = df["loan_amount"] / (df["loan_to_value_ratio"] / 100)
    df["value_gap"]        = df["implied_value"] / df["property_value"]
    df["monthly_principal"] = df["loan_amount"] / df["loan_term"]      # loan_term en mois
    df["pti_proxy"]        = df["monthly_principal"] / (inc / 12)      # proxy du payment-to-income
    df["debt_amount"]      = df["debt_to_income_ratio"] * inc / 100    # dette mensuelle/annuelle implicite
    df["residual_income"]  = inc - df["loan_amount"] / (df["loan_term"] / 12)
    
    df["n_missing"] = df.isna().sum(axis=1)
    return df.replace([np.inf, -np.inf], np.nan)

data_train_feat = add_features(data_train)
data_test_feat = add_features(data_test)

# 2) Pour chaque colonne numérique : conversion, indicateur de manquant, médiane
new_cols = ["income_vs_msa", "loan_to_income", "pti_proxy", "debt_amount", "residual_income"]

for col in new_cols:
    data_train_feat[col + "_missing"] = data_train_feat[col].isnull().astype(int)  # 1) on note les manquants
    data_train_feat[col] = data_train_feat[col].fillna(data_train_feat[col].median())   # 2) on remplit

for col in new_cols:
    data_test_feat[col + "_missing"] = data_test_feat[col].isnull().astype(int) 
    data_test_feat[col] = data_test_feat[col].fillna(data_train_feat[col].median()) 

#"""""""""""""""""""""""""""""""""""""""""""""""" SEPARATION DES COVARIABLES ET DE LA VARIABLE CIBLE DANS LES DONNÉES D'ENTRAÎNEMENT """"""""""""""""""""""""""""""""""""""""

X_train = data_train_feat.drop(columns=["accepted", "id"])
Y_train = data_train_feat["accepted"]
X_test = data_test_feat.drop(columns=["id"])

#"""""""""""""""""""""""""""""""""""""""""""""""" NORMALISATION DES DONNEES SI BESOIN """"""""""""""""""""""""""""""""""""""

scaler = StandardScaler()
X_train_normed = scaler.fit_transform(X_train)
X_test_normed = scaler.transform(X_test)

#""""""""""""""""""""""""""""""""""""""""""""""" CONVERSION EN DATASET ET EN DATALOADER """"""""""""""""""""""""""""""""""""""""""""""""


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)            

    def __getitem__(self, idx):
        if self.y is None:            
            return self.X[idx]
        return self.X[idx], self.y[idx]


train_ds = TabularDataset(X_train_normed, Y_train)
test_ds  = TabularDataset(X_test_normed)          

train_loader = DataLoader(train_ds, batch_size=1000, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False)


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_normed, Y_train, test_size=0.2, stratify=Y_train, random_state=42
)
val_loader = DataLoader(TabularDataset(X_val), batch_size=1024, shuffle=False)

def objective(trial):
    input_shape = X_train_normed.shape[1]
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    p          = trial.suggest_float("p", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    h1 = trial.suggest_categorical("h1", [64, 128, 256, 512])
    h2 = trial.suggest_categorical("h2", [32, 64, 128, 256])
    h3 = trial.suggest_categorical("h3", [16, 32, 64, 128])
    h4 = trial.suggest_categorical("h4", [8, 16, 32, 64])

    model = MLP_CE(input_shape, h1, h2, h3, h4, 2, p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(TabularDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)

    best_auc, best_epoch = 0, 0
    for epoch in range(30):
        train_loop_CE(train_loader, model, 1, optimizer, criterion, lr, device)
        scores = running_classifier(val_loader, model, device)
        auc = roc_auc_score(y_val, scores)

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1

        trial.report(auc, epoch)
        if trial.should_prune():          
            raise optuna.TrialPruned()

    trial.set_user_attr("best_epoch", best_epoch)
    return best_auc

study = optuna.create_study(direction="maximize",
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=50)

print(study.best_value)
print(study.best_params)
print(study.best_trial.user_attrs["best_epoch"])


# 0.8685812942698062
#{'lr': 0.00032224714117548976, 'weight_decay': 2.1498645119039145e-05, 'p': 0.12611115086159516, 'batch_size': 256, 'h1': 128, 'h2': 64, 'h3': 128, 'h4': 64}
#25

# AVEC CLEAN ET ADD FEATURES :

# 0.9014081401760269
# {'lr': 0.00034998107070177746, 'weight_decay': 2.60507560886816e-05, 'p': 0.21999509297292325, 'batch_size': 256, 'h1': 128, 'h2': 128, 'h3': 128, 'h4': 32}
# 23

