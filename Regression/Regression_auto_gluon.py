import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from tabicl import TabICLClassifier
from autogluon.tabular import TabularPredictor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_train = pd.read_csv("data_train_clean.csv")
data_test = pd.read_csv("data_test_clean.csv")

data_train = data_train.drop(columns=["Unnamed: 0"])
data_test = data_test.drop(columns=["Unnamed: 0"])


X_train = data_train.drop(columns=["TotalKg", "id"])
Y_train = data_train["TotalKg"]
X_test = data_test.drop(columns=["id"])



predictor = TabularPredictor(
    label="TotalKg",
    eval_metric="root_mean_squared_error",       # adapte à la métrique de ta compétition
).fit(
    data_train.drop(columns=["id"]),
    presets="best",
    time_limit=300,             # secondes, à ajuster
    num_gpus=1,                  # optionnel : "auto" par défaut
)

y_pred = predictor.predict(data_test.drop(columns=["id"]))

submission = pd.DataFrame({
    "id": data_test["id"],
    "TotalKg": y_pred    # colonne de la classe positive (c'est un DataFrame ici)
})

submission.to_csv("submission_regression_auto_gluon.csv", index=False)