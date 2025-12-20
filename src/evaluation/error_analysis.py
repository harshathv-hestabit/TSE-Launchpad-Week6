import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"
RESULTS_DIR = BASE_DIR / "results"

EVAL_DIR.mkdir(exist_ok=True)


def load_artifacts():
    model_pipeline = joblib.load(MODEL_DIR / "best_model_tuned.pkl")

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    return model_pipeline, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    )
    plt.tight_layout()

    filename = (
        "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    )
    plt.savefig(RESULTS_DIR / filename, dpi=200)
    plt.close()


def error_clustering(X, y_true, y_pred):
    errors = X.copy()
    errors["true_label"] = y_true
    errors["predicted_label"] = y_pred
    errors["is_error"] = y_true != y_pred

    error_summary = (
        errors[errors["is_error"]]
        .groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    error_summary.to_csv(
        RESULTS_DIR / "error_clusters.csv",
        index=False
    )

def save_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(RESULTS_DIR / "classification_report.csv")

def main():
    model_pipeline, X_test, y_test = load_artifacts()

    y_pred = model_pipeline.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, normalize=False)
    plot_confusion_matrix(y_test, y_pred, normalize=True)

    error_clustering(X_test, y_test, y_pred)
    save_classification_report(y_test, y_pred)

    print("Error analysis completed")

if __name__ == "__main__":
    main()