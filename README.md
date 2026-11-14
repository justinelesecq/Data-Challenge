# Data Challenge 
Ce projet universitaire consiste en deux tâches d'apprentissage supervisé distinctes. 

La première est une tâche de classification binaire ayant pour objectif de prédire si une demande de prêt immobilier est accordée (`accepted = 1`) ou refusée (`accepted = 0`).

La seconde est une tâche de régression permettant de prédire les performances d'un athlète en powerlifting.

## Installation

```bash
git clone <url-du-depot>
```

## Structure du projet

Les dossiers `Classification` et `Regression` comportent toutes les études et tous les essais réalisés durant la phase de réalisation de ce projet.

Les dossiers `Classification_finale` et `Regression_finale` contiennent les codes permettant de reproduire l'apprentissage et la prédiction pour les soumissions retenues.

Plus précisément, les codes composant le dossier `Classification_finale` doivent se lire dans l'ordre suivant :

- `Classification_code_final_p1.ipynb` : pré-traitement sur les données, régression logistique, arbre de décision, gradient boosting, SVM, AutoGluon
- `Classification_code_final_p2.ipynb` : étude approfondie sur le gradient boosting
- `Classification_MLP_arcitecture` : architecture du MLP
- `Classification_MLP_clean` : entraînement et prédictions du MLP
- `Classification_MLP_opt` : Recherche d'hyperparamètres du MLP avec `Optuna`

Le dossier `Regression_finale` ne contient qu'un seul code `Regression_code_final.ipynb` comportant le pré-traitement sur les données et tous les modèles essayés.

## Données

Les données ne sont pas incluses dans le dépôt. Il faut les télécharger depuis Kaggle :

- Classification (https://www.kaggle.com/t/2f3b2035f93e49cdbe9b64732b8c0408) : placer `train.csv` et `test.csv` dans le dossier `Classification_finale/`
- Régression (https://www.kaggle.com/t/8302b2de746f4d0992cff7722a83ceeb) : placer `train.csv` et `test.csv` dans le dossier `Regression_finale/`

