import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb, optuna
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, r2_score, mean_absolute_error)
from sklearn.pipeline import make_pipeline


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_train = pd.read_csv("data_train_clean.csv")
data_test = pd.read_csv("data_test_clean.csv")

data_train = data_train.drop(columns=["Unnamed: 0"])
data_test = data_test.drop(columns=["Unnamed: 0"])


X_train = data_train.drop(columns=["TotalKg", "id"])
Y_train = data_train["TotalKg"]
X_test = data_test.drop(columns=["id"])


# LIGHTGBM (gradient boosting)


# model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06, num_leaves=63)
# model.fit(X_train, Y_train)

# y_pred = model.predict(X_test)

# submission = pd.DataFrame({
#     "id": data_test["id"],
#     "TotalKg": y_pred   
# })
# submission.to_csv("regression_submission_lgb_random.csv", index=False)


# XGBoost et optuna : 


dtrain = xgb.DMatrix(X_train, label=Y_train)

DEVICE = "cuda"   # mettre "cpu" si tu n'as pas de GPU

def objective(trial):
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "tree_method": "hist", "device": DEVICE,
        "eta": 0.05,
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }
    cv = xgb.cv(params, dtrain, num_boost_round=5000, nfold=5, seed=0,
                early_stopping_rounds=100, verbose_eval=False)
    trial.set_user_attr("best_iter", len(cv))
    return cv["test-rmse-mean"].iloc[-1]

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)

print("Meilleur RMSE (CV) :", study.best_value)
print("Meilleurs paramètres :", study.best_params)

# Modèle final sur tout le train
best = study.best_params | {"objective": "reg:squarederror", "eval_metric": "rmse",
                            "tree_method": "hist", "device": DEVICE, "eta": 0.05}
n_iter = int(study.best_trial.user_attrs["best_iter"] * 1.1)
final = xgb.train(best, dtrain, num_boost_round=n_iter)

y_pred = final.predict(xgb.DMatrix(X_test))

submission = pd.DataFrame({
    "id": data_test["id"],
    "TotalKg": y_pred
})
submission.to_csv("regression_submission_xgb_optuna.csv", index=False)


# Meilleur RMSE (CV) : 78.3029402992892
# Meilleurs paramètres : {'max_depth': 8, 'min_child_weight': 3.247724675235288, 'subsample': 0.9359283588320708, 'colsample_bytree': 0.5611221319434191, 'reg_lambda': 0.0351522734357646, 'reg_alpha': 0.024447397056311403}
