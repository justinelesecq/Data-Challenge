#"""""""""""""""""""""""""""""""""""""""""" IMPORT DES LIBRAIRIES """"""""""""""""""""""""""""""""""""""""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from xgboost import XGBClassifier


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


#"""""""""""""""""""""""""""""""""""""""""""""""""""" BDT """""""""""""""""""""""""""""""""""""""""""""""""""

# On doit régler 3 paramètres : `n_estimators`, `learning_rate` et `max_depth` :

# Pas de n_jobs=-1 avec le GPU : on laisse le GPU faire le travail,
# et on lance les combinaisons du grid une par une
grid = GridSearchCV(
    XGBClassifier(
        device="cuda",          # utilise le GPU
        tree_method="hist",     # méthode rapide, compatible GPU
        random_state=42,
        eval_metric="auc",
    ),
    param_grid={
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.05, 0.1, 0.3],
    },
    scoring="roc_auc",
    cv=5,
    n_jobs=1,
)
grid.fit(X_train, Y_train)
print(grid.best_params_, grid.best_score_)

# Entraînement final et prédictions
best_model = grid.best_estimator_      # déjà réentraîné sur tout le train
y_proba = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": data_test["id"], "accepted": y_proba})
submission.to_csv("submission_xgb_clean.csv", index=False)

# ENTRAINEMENT FINAL :

# Apprenant faible : un arbre peu profond (souvent une "souche", max_depth=1)
# base = DecisionTreeClassifier(max_depth=2)

# model_adaboost = AdaBoostClassifier(
#     estimator=base,        # s'appelait base_estimator avant sklearn 1.2
#     n_estimators=200,
#     learning_rate=0.5,
#     random_state=42,
# )

# model_adaboost.fit(X_train, y_train)

# # PREDICTIONS :

# # Probabilité de la classe 1 (accepté)
# y_proba = model_svm.predict_proba(X_test_normed)[:, 1]

# submission = pd.DataFrame({
#     "id": data_test["id"],
#     "accepted": y_proba
# })

# submission.to_csv("submission_svm.csv", index=False)

