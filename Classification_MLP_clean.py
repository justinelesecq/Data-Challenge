import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from Classification_MLP_architecture import MLP_CE, train_loop_CE, running_classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#"""""""""""""""""""""""""""""""""""""" CHARGEMENTS DES DONNÉES D'ENTREE ET DE TEST """"""""""""""""""""""""

data_train = pd.read_csv("data_train_clean.csv")
data_test = pd.read_csv("data_test_clean.csv")

data_train = data_train.drop(columns=["Unnamed: 0"])
data_test = data_test.drop(columns=["Unnamed: 0"])

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

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)


model_CE = MLP_CE(
        input_shape=X_train_normed.shape[1],
        hidden_1=128,
        hidden_2=64,
        hidden_3=128,
        hidden_4=64,
        output_shape=2,
        p=0.12611115086159516,
        activation=torch.nn.ReLU(),
    ).float()

optimizer_CE = torch.optim.Adam(model_CE.parameters(), lr=0.00032224714117548976, weight_decay= 2.1498645119039145e-05)

criterion = torch.nn.CrossEntropyLoss().to(device)

model_trained = train_loop_CE(trainloader=train_loader, model = model_CE, n_epochs=25, optimizer=optimizer_CE, criterion=criterion, lr=0.00032224714117548976, device=device)
torch.save(model_trained.state_dict(), "MLP_1")

print("c'est fini de s'entrainer")

y_proba = running_classifier(testloader=test_loader, model=model_trained, device=device)

submission = pd.DataFrame({
    "id": data_test["id"],
    "accepted": y_proba
})
submission.to_csv("submission_MLP_optuna_clean.csv", index=False)

print("c'est tout soumis")


# Optimisation du MLP : use Xavier weights dans train_loop, hyperparameters, automl (DRAGON)

# MLP_1 : epochs = 5, h1=256, h2=128, h3=64, h4=32, p=0, lr=1e-3, batch_size=200
# MLP_2 : epochs = 10, h1=256, h2=128, h3=64, h4=32, p=0, lr=1e-3, batch_size=1000
# MLP_optuna : epochs = 25, h1=128, h2=64, h3=128, h4=64, p=0.12611115086159516, lr=0.00032224714117548976, batch_size=256, weight_decay= 2.1498645119039145e-05
# MLP_optuna_clean : score kaggle = 0.90821

