import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from Classification_MLP_architecture import MLP_CE, train_loop_CE, running_classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#"""""""""""""""""""""""""""""""""""""" CHARGEMENTS DES DONNÉES D'ENTREE ET DE TEST """"""""""""""""""""""""

data_train = pd.read_csv("data_train_clean.csv")
data_test = pd.read_csv("data_test_clean.csv")

data_train = data_train.drop(columns=["Unnamed: 0"])
data_test = data_test.drop(columns=["Unnamed: 0"])

#"""""""""""""""""""""""""""""""""""""""""""""""""""" ADD FEATURES """"""""""""""""""""""""""""""""""""""""""""""""

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

#"""""""""""""""""""""""""""""""""""""""""""""" On enlève les NaN des nouvelles variables """"""""""""""""""""""""""""""""""""

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
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)


model_CE = MLP_CE(
        input_shape=X_train_normed.shape[1],
        hidden_1=128,
        hidden_2=128,
        hidden_3=128,
        hidden_4=32,
        output_shape=2,
        p=0.21999509297292325,
        activation=torch.nn.ReLU(),
    ).float()

optimizer_CE = torch.optim.Adam(model_CE.parameters(), lr=0.00034998107070177746, weight_decay=2.60507560886816e-05)

criterion = torch.nn.CrossEntropyLoss().to(device)

model_trained = train_loop_CE(trainloader=train_loader, model = model_CE, n_epochs=23, optimizer=optimizer_CE, criterion=criterion, lr=0.00034998107070177746, device=device)
torch.save(model_trained.state_dict(), "MLP_1")

print("c'est fini de s'entrainer")

y_proba = running_classifier(testloader=test_loader, model=model_trained, device=device)

submission = pd.DataFrame({
    "id": data_test["id"],
    "accepted": y_proba
})
submission.to_csv("submission_MLP_opt_clean_addfeatures.csv", index=False)

print("c'est tout soumis")


# MLP_1 : epochs = 5, h1=256, h2=128, h3=64, h4=32, p=0, lr=1e-3, batch_size=200
# MLP_2 : epochs = 10, h1=256, h2=128, h3=64, h4=32, p=0, lr=1e-3, batch_size=1000
# MLP_optuna_noclean : epochs = 25, h1=128, h2=64, h3=128, h4=64, p=0.12611115086159516, lr=0.00032224714117548976, batch_size=256, weight_decay= 2.1498645119039145e-05

# MLP_optuna_clean_addfeat : {'lr': 0.00034998107070177746, 'weight_decay': 2.60507560886816e-05, 'p': 0.21999509297292325, 'batch_size': 256, 'h1': 128, 'h2': 128, 'h3': 128, 'h4': 32}
# epochs 23

