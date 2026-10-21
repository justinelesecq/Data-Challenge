# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""""""""""""" MLP """""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch.nn as nn
import torch
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP_CE(nn.Module):
    """
    MLP with 3 hidden layers and a dropout layer.
    The activation function used is a ReLU.
    This MLP can classify if a seed is true or false (one seed at a time).

    Args:
        input_shape: number of features for 3 hits (one seed)
        hidden_1: number of neurons for the first hidden layer
        hidden_2: number of neurons for the second hidden layer
        hidden_3: number of neurons for the third hidden layer
        output_shape: 2 (0:False, 1:True)
        p: dropout coefficient
        activation: activation function (ReLU)

    Returns:
        Weither the seed is true (1) or false (0).


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


# """"""""""""""""""""""""""""""""""""""""""""" TRAINING """""""""""""""""""""""""""""""""""""""""


def train_loop_CE(trainloader, model: MLP_CE, n_epochs: int, optimizer, criterion, lr: float, device=device):
    """
    Returns the trained model.
    """
    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):

        print("epoch:", epoch)

        for i, (X, y) in enumerate(trainloader):
            inputs = X.to(device)
            labels = y.to(device).float().unsqueeze(1)

            # Gradients to zero
            optimizer.zero_grad()
            # Compute prediction and loss
            pred = model(inputs)
            loss = criterion(pred, labels.squeeze().long())

            # Update gradients and Update model's weights:
            loss.backward()
            optimizer.step()

    return model


#"""""""""""""""""""""""""""""""""""""""""""""""""" TESTING """""""""""""""""""""""""""""""""""""""

def running_classifier(testloader, model, device=device):
    """
    returns signal = true seeds found by the classifier
    """

    model.eval()

    scores = []

    with torch.no_grad():
        for X in testloader:
            X = X.to(device)
            pred = model(X) 
            proba = torch.softmax(pred, dim=1)[:, 1] # probability for each class to be the right one.
            scores.append(proba.cpu())

    scores = torch.cat(scores).numpy()

    return scores