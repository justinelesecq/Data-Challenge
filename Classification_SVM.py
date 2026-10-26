#"""""""""""""""""""""""""""""""""""""""""" IMPORT DES LIBRAIRIES """"""""""""""""""""""""""""""""""""""""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from cuml.svm import SVC   # à la place de sklearn.svm.SVC

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

model_svm = SVC(kernel="rbf", C=0.8179499475211672, gamma=0.012561043700013555, class_weight="balanced", probability=True)
model_svm.fit(X_train_normed, Y_train)

# PREDICTIONS :

# Probabilité de la classe 1 (accepté)
y_proba = model_svm.predict_proba(X_test_normed)[:, 1]

submission = pd.DataFrame({
    "id": data_test["id"],
    "accepted": y_proba
})

submission.to_csv("submission_svm.csv", index=False)

