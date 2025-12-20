import json
import joblib
import optuna
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
TUNING_DIR = BASE_DIR / "tuning"

MODEL_DIR.mkdir(exist_ok=True)
TUNING_DIR.mkdir(exist_ok=True)

def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    return X_train, y_train

def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        f1 = f1_score(y_val, y_pred, average="weighted")
        f1_scores.append(f1)

    return np.mean(f1_scores)


def main():
    X, y = load_data()

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_multiclass_tuning"
    )

    study.optimize(lambda trial: objective(trial, X, y), n_trials=30)

    best_params = study.best_params
    best_score = study.best_value

    best_model = XGBClassifier(
        **best_params,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_model)
    ])

    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, MODEL_DIR / "best_model_tuned.pkl")

    results = {
        "best_params": best_params,
        "best_cv_f1_weighted": best_score,
        "n_trials": len(study.trials)
    }

    with open(TUNING_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Tuning completed")
    print("Best CV F1 (weighted):", best_score)


if __name__ == "__main__":
    main()