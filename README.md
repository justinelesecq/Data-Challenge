# Data Challenge 
Ce dépôt contient un projet universitaire consistant en deux tâches d'apprentissage supervisé distinctes. 

La première est une tâche de classification binaire ayant pour objectif de prédire si une demande de prêt immobilier est accordée (`accepted = 1`) ou refusée (`accepted = 0`).

La seconde est une tâche de régression permettant de prédire les performances d'un athlète en powerlifting.

Projet réalisé dans le cadre du Master 2 MIA, module « Data Challenge et apprentissage supervisé ».

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) pour gérer l'environnement Python et les dépendances.

1. Installer uv (si ce n'est pas déjà fait) :

```bash
pip install uv
```

2. Cloner le dépôt :

```bash
git clone https://github.com/justinelesecq/Data-Challenge.git
cd Data-Challenge
```

3. Créer l'environnement et installer les dépendances (versions exactes de `uv.lock`, et version de Python de `.python-version`) :

```bash
uv sync
```

4. Lancer Jupyter avec cet environnement :

```bash
uv run jupyter notebook
```

Par exemple, pour exécuter un script Python :

```bash
uv run python Classification_finale/Classification_MLP_opt.py
```

## Structure du projet

Les dossiers `Classification` et `Regression` comportent toutes les études et tous les essais menés au cours de ce projet.

Les dossiers `Classification_finale` et `Regression_finale` contiennent les codes permettant de reproduire l'apprentissage et la prédiction pour les soumissions retenues.

Plus précisément, les codes composant le dossier `Classification_finale` doivent se lire dans l'ordre suivant :

- `Classification_code_final_p1.ipynb` : pré-traitement sur les données, régression logistique, arbre de décision, gradient boosting, SVM, AutoGluon
- `Classification_code_final_p2.ipynb` : étude approfondie sur le gradient boosting
- `Classification_MLP_architecture.py` : architecture du MLP
- `Classification_MLP_clean.py` : entraînement et prédictions du MLP
- `Classification_MLP_opt.py` : recherche d'hyperparamètres du MLP avec `Optuna`

Le dossier `Regression_finale` ne contient qu'un seul code `Regression_code_final.ipynb` comportant le pré-traitement sur les données et tous les modèles essayés.

## Données

Les données ne sont pas incluses dans le dépôt. Il faut les télécharger depuis Kaggle :

- [Challenge de classification](https://www.kaggle.com/t/2f3b2035f93e49cdbe9b64732b8c0408) : placer `train.csv` et `test.csv` dans le dossier `Classification_finale/`
- [Challenge de régression](https://www.kaggle.com/t/8302b2de746f4d0992cff7722a83ceeb) : placer `train.csv` et `test.csv` dans le dossier `Regression_finale/`

## Auteur

- Justine LESECQ

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

