#"""""""""""""""""""""""""""""""""""""""""" IMPORT DES LIBRAIRIES """"""""""""""""""""""""""""""""""""""""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.impute import SimpleImputer
from cuml.svm import SVC   # à la place de sklearn.svm.SVC

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



#"""""""""""""""""""""""""""""""""""""""""""""""""""" SVM A NOYAU """""""""""""""""""""""""""""""""""""""""""""""""""

# La normalisation est obligatoire. Contrairement aux arbres, le SVM repose sur des distances.

# ENTRAINEMENT FINAL :

model_svm = SVC(
    kernel="rbf",
    C=0.8179499475211672,
    gamma=0.012561043700013555,
    class_weight="balanced",
    probability=True,
)
model_svm.fit(X_train_normed, Y_train)

y_proba = model_svm.predict_proba(X_test_normed)[:, 1]

# model_svm = SVC(kernel="rbf", C=0.8179499475211672, gamma=0.012561043700013555, class_weight="balanced", probability=True)
# model_svm.fit(X_train_normed, Y_train)

# # PREDICTIONS :

# # Probabilité de la classe 1 (accepté)
# y_proba = model_svm.predict_proba(X_test_normed)[:, 1]

submission = pd.DataFrame({
    "id": data_test["id"],
    "accepted": y_proba
})

submission.to_csv("submission_svm.csv", index=False)

