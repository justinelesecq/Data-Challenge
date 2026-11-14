# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""""""""" ARCHITECTURE DU MLP """""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Imports des librairies :

import torch.nn as nn
import torch
import pandas as pd

# Utilisation de GPU :

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP_CE(nn.Module):
    """
    Perceptron multicouche (MLP) pour la classification, entraîné avec une CrossEntropyLoss.

    Arguments :
    
        input_shape (int) : nombre de variables en entrée
        hidden_1 (int)    : nombre de neurones de la 1re couche cachée
        hidden_2 (int)    : nombre de neurones de la 2e couche cachée
        hidden_3 (int)    : nombre de neurones de la 3e couche cachée
        hidden_4 (int)    : nombre de neurones de la 4e couche cachée
        output_shape (int): nombre de classes en sortie (2 : refusé / accepté)
        p (float)         : probabilité de dropout, entre 0 et 1 (fraction de neurones
                            désactivés aléatoirement à l'entraînement, contre le
                            sur-apprentissage)
        activation        : fonction d'activation appliquée après chaque couche cachée
                            (par défaut `torch.nn.ReLU()`)
    """

    def __init__(
        self,
        input_shape: int,
        hidden_1: int,
        hidden_2: int,
        hidden_3: int,
        hidden_4: int,
        output_shape: int,
        p: float,
        activation=torch.nn.ReLU(),
    ):
        super(MLP_CE, self).__init__()
        self.W1 = torch.nn.Linear(input_shape, hidden_1)
        self.activation = activation
        self.W2 = torch.nn.Linear(hidden_1, hidden_2)
        self.W3 = torch.nn.Linear(hidden_2, hidden_3)
        self.W4 = torch.nn.Linear(hidden_3, hidden_4)
        self.dropout = torch.nn.Dropout(p)
        self.output_layer = torch.nn.Linear(hidden_4, output_shape)

    def forward(self, x):
        out = self.W1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.W2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.W3(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.W4(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out


# """"""""""""""""""""""""""""""""""""""""""""" FONCTION DE TRAINING """""""""""""""""""""""""""""""""""""""""


def train_loop_CE(trainloader, model: MLP_CE, n_epochs: int, optimizer, criterion, lr: float, device=device):
    """
    Entraîne le modèle sur les données d'entraînement.

    Arguments :
        trainloader (DataLoader) : itérateur sur les batchs d'entraînement ; chaque
                                   élément est un couple (X, y) avec X les covariables et
                                   y les étiquettes
        model (MLP_CE)           : modèle à entraîner 
        n_epochs (int)           : nombre de passages complets sur le jeu d'entraînement
        optimizer                : optimiseur PyTorch
        criterion                : fonction de perte (ici : `nn.CrossEntropyLoss()`)
        lr (float)               : taux d'apprentissage
        device                   : CPU ou GPU

    Retourne :
        MLP_CE : le modèle entraîné 
    """
    
    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):

        print("epoch:", epoch)

        for i, (X, y) in enumerate(trainloader):
            inputs = X.to(device)
            labels = y.to(device).float().unsqueeze(1)

            # Initialisation des gradients à zero :
            optimizer.zero_grad()
            
            pred = model(inputs)
            loss = criterion(pred, labels.squeeze().long())

            # Mise à jour des gradients et des poids :
            loss.backward()
            optimizer.step()

    return model


# """""""""""""""""""""""""""""""""""""""""""""""""" FONCTION DE TEST """""""""""""""""""""""""""""""""""""""

def running_classifier(testloader, model, device=device):
    """
    Calcule, pour chaque observation du jeu de test, la probabilité d'appartenir
    à la classe 1 (demande acceptée)

    Arguments :
        testloader (DataLoader) : itérateur sur les batchs de test ; chaque élément est
                                  directement un tenseur X de covariables (sans étiquettes)
        model (MLP_CE)          : modèle entraîné
        device                  : CPU ou GPU

    Retourne :
        numpy.ndarray : tableau 1D de taille (nombre d'observations de test) contenant
        les probabilités de la classe 1 (valeurs entre 0 et 1), dans l'ordre du testloader
    """
    
    model.eval()

    scores = []

    with torch.no_grad():
        for X in testloader:
            X = X.to(device)
            pred = model(X) 
            proba = torch.softmax(pred, dim=1)[:, 1] 
            scores.append(proba.cpu())

    scores = torch.cat(scores).numpy()

    return scores