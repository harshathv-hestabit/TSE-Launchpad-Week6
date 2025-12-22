import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from src.utils.logger import logTool

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

logger = logTool(__name__)

def load_data():
    X_train = pd.read_csv(BASE_DIR / "data/processed/X_train.csv")
    y_train = pd.read_csv(BASE_DIR / "data/processed/y_train.csv").squeeze()

    X_test = pd.read_csv(BASE_DIR / "data/processed/X_test.csv")
    y_test = pd.read_csv(BASE_DIR / "data/processed/y_test.csv").squeeze()
    
    return X_train, y_train, X_test, y_test


def get_models():
    return {
        "logistic_regression": OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs"
            )
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.0001,
            max_iter=300,
            random_state=42
        )
    }


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    }


def cross_validate(model, X, y, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fold_metrics = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_tr, y_tr)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)

        fold_metrics.append(
            compute_metrics(y_val, y_pred, y_prob)
        )

    return pd.DataFrame(fold_metrics).mean().to_dict()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    X_train, y_train, X_test, y_test = load_data()
    models = get_models()

    cv_metrics = {}
    best_model_name = None
    best_score = -np.inf

    for name, model in models.items():
        logger.info(f"Running CV for model: {name}")
        metrics = cross_validate(model, X_train, y_train)
        cv_metrics[name] = metrics

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = name

    best_model = models[best_model_name]

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_model)
    ])

    final_pipeline.fit(X_train, y_train)

    y_test_pred = final_pipeline.predict(X_test)
    y_test_prob = final_pipeline.predict_proba(X_test) 

    test_metrics = compute_metrics(
        y_test,
        y_test_pred,
        y_test_prob
    )
    
    classes = sorted(y_test.unique())
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        classes,
        RESULTS_DIR / "confusion_matrix.png"
    )

    joblib.dump(
        final_pipeline,
        MODEL_DIR / "best_model.pkl"
    )

    with open(EVAL_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics
            },
            f,
            indent=4
        )

    logger.info(f"Best model saved: {best_model_name}")

if __name__ == "__main__":
    main()