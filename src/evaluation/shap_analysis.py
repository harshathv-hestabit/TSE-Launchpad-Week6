import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"
RESULTS_DIR = BASE_DIR / "results"

EVAL_DIR.mkdir(exist_ok=True)

def load_artifacts():
    model_pipeline = joblib.load(MODEL_DIR / "best_model_tuned.pkl")

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

    return model_pipeline, X_train, y_train

def run_shap_analysis(model_pipeline, X):
    scaler = model_pipeline.named_steps["scaler"]
    xgb_model = model_pipeline.named_steps["model"]

    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    X_sample_scaled = scaler.transform(X_sample)

    X_sample_scaled_df = pd.DataFrame(
        X_sample_scaled,
        columns=X.columns,
        index=X_sample.index
    )

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample_scaled_df)

    if isinstance(shap_values, list):
        for class_idx, class_shap in enumerate(shap_values):
            shap.summary_plot(
                class_shap,
                X_sample_scaled_df,
                feature_names=X.columns,
                show=False
            )
            plt.savefig(
                RESULTS_DIR / f"shap_summary_class_{class_idx}.png",
                dpi=200
            )
            plt.clf()

    else:
        n_classes = shap_values.shape[2]

        for class_idx in range(n_classes):
            shap.summary_plot(
                shap_values[:, :, class_idx],
                X_sample_scaled_df,
                feature_names=X.columns,
                show=False
            )
            plt.savefig(
                RESULTS_DIR / f"shap_summary_class_{class_idx}.png",
                dpi=200
            )
            plt.clf()

def plot_feature_importance(model_pipeline, X):
    xgb_model = model_pipeline.named_steps["model"]

    importance = xgb_model.feature_importances_
    feature_names = X.columns

    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

def main():
    model_pipeline, X_train, _ = load_artifacts()

    run_shap_analysis(model_pipeline, X_train)
    plot_feature_importance(model_pipeline, X_train)

    print("SHAP analysis completed")


if __name__ == "__main__":
    main()