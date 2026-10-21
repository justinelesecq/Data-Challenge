import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from Classification_MLP_architecture import MLP_CE, train_loop_CE, running_classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#"""""""""""""""""""""""""""""""""""""" CHARGEMENTS DES DONNÉES D'ENTREE ET DE TEST """"""""""""""""""""""""

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#"""""""""""""""""""""""""""""""""""""" TRAITEMENT DES DONNÉES """""""""""""""""""""""""""""""""""""""""""""

# Concernant la variable co-applicant_age :
data_train["has_coapplicant"] = data_train["co-applicant_age"].notnull().astype(int)
data_test["has_coapplicant"] = data_test["co-applicant_age"].notnull().astype(int)

# Concernant les variables continues :
# 1) Tranches en texte -> valeurs numériques (milieu de chaque tranche)
dti_tranches = {
    "<20%": 15, "20%-<30%": 25, "30%-<36%": 33,
    "50%-60%": 55, ">60%": 65,
}

data_train["debt_to_income_ratio"] = data_train["debt_to_income_ratio"].replace(dti_tranches)
data_test["debt_to_income_ratio"] = data_test["debt_to_income_ratio"].replace(dti_tranches)

for df in (data_train, data_test):
    df["debt_to_income_ratio"] = pd.to_numeric(df["debt_to_income_ratio"], errors="coerce")

# 2) Pour chaque colonne numérique : conversion, indicateur de manquant, médiane
num_cols = ["income", "property_value", "loan_term", "loan_to_value_ratio",
            "debt_to_income_ratio", "total_units"]

for col in num_cols:
    data_train[col + "_missing"] = data_train[col].isnull().astype(int)  # 1) on note les manquants
    data_train[col] = data_train[col].fillna(data_train[col].median())   # 2) on remplit

for col in num_cols:
    data_test[col + "_missing"] = data_test[col].isnull().astype(int) 
    data_test[col] = data_test[col].fillna(data_train[col].median())  

# Concernant les variables catégorielles :
# 1) Colonnes numériques qui sont en fait des codes
code_cols = ["county_code", "applicant_ethnicity-1", "co-applicant_ethnicity-1",
             "applicant_race-1", "co-applicant_race-1"]

# 2) Toutes les colonnes de texte, détectées automatiquement
text_cols = data_train.select_dtypes(exclude="number").columns.tolist()

# 3) Liste finale de toutes les catégorielles, sans doublons
cat_cols = list(dict.fromkeys(text_cols + code_cols))
print(cat_cols)

# 4) NaN -> "Inconnu", puis tout en texte
for df in (data_train, data_test):
    df[cat_cols] = df[cat_cols].astype(object).fillna("Inconnu").astype(str)

# 5) Encodage numérique
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
data_train[cat_cols] = encoder.fit_transform(data_train[cat_cols])
data_test[cat_cols] = encoder.transform(data_test[cat_cols])

print(data_train.dtypes.value_counts())  # doit n'afficher que int64 / float64


#"""""""""""""""""""""""""""""""""""""""""""""""" SEPARATION DES COVARIABLES ET DE LA VARIABLE CIBLE DANS LES DONNÉES D'ENTRAÎNEMENT """"""""""""""""""""""""""""""""""""""""

X_train = data_train.drop(columns=["accepted", "id"])
Y_train = data_train["accepted"]
X_test = data_test.drop(columns=["id"])

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
        return len(self.X)            # nombre d'exemples

    def __getitem__(self, idx):
        if self.y is None:            # cas du test (pas de cible)
            return self.X[idx]
        return self.X[idx], self.y[idx]


train_ds = TabularDataset(X_train_normed, Y_train)
test_ds  = TabularDataset(X_test_normed)          # sans y

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
        if trial.should_prune():          # abandonne les essais mauvais
            raise optuna.TrialPruned()

    trial.set_user_attr("best_epoch", best_epoch)
    return best_auc

study = optuna.create_study(direction="maximize",
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=50)

print(study.best_value)
print(study.best_params)
print(study.best_trial.user_attrs["best_epoch"])


# Optimisation du MLP : use Xavier weights dans train_loop, hyperparameters, automl (DRAGON)

# 0.8685812942698062
#{'lr': 0.00032224714117548976, 'weight_decay': 2.1498645119039145e-05, 'p': 0.12611115086159516, 'batch_size': 256, 'h1': 128, 'h2': 64, 'h3': 128, 'h4': 64}
#25
