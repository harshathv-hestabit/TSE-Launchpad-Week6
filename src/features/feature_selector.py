from pathlib import Path
from src.utils.logger import logTool

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

DATA_DIR = Path("src/data/processed")
FEATURE_DIR = Path("src/features")
RESULTS_DIR = Path("src/results")

X_TRAIN_FILE = DATA_DIR / "X_train.csv"
Y_TRAIN_FILE = DATA_DIR / "y_train.csv"

SELECTED_FEATURES_FILE = RESULTS_DIR / "selected_features.json"

CORR_THRESHOLD = 0.95
MI_TOP_K = 30
RFE_TOP_K = 25
RANDOM_STATE = 42

logger = logTool(__name__)

def load_features() -> tuple[pd.DataFrame, pd.Series]:
    if not X_TRAIN_FILE.exists() or not Y_TRAIN_FILE.exists():
        raise FileNotFoundError("Training feature files not found")

    logger.info("Loading training features")
    X = pd.read_csv(X_TRAIN_FILE)
    y = pd.read_csv(Y_TRAIN_FILE).squeeze()

    return X, y


def correlation_filter(X: pd.DataFrame) -> list[str]:
    logger.info("Applying correlation-based feature filtering")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > CORR_THRESHOLD)
    ]

    selected = [c for c in X.columns if c not in to_drop]
    logger.info(f"Features after correlation filter: {len(selected)}")

    return selected


def mutual_information_filter(X: pd.DataFrame, y: pd.Series) -> list[str]:
    logger.info("Applying mutual information (classification)")

    mi_scores = mutual_info_classif(
        X,
        y,
        random_state=RANDOM_STATE,
        discrete_features=False,
    )

    mi_series = pd.Series(mi_scores, index=X.columns)

    selected = (
        mi_series.sort_values(ascending=False)
        .head(MI_TOP_K)
        .index.tolist()
    )

    logger.info(f"Top MI-selected features: {len(selected)}")
    return selected


def rfe_filter(X: pd.DataFrame, y: pd.Series) -> list[str]:
    logger.info("Applying Recursive Feature Elimination (classification)")

    estimator = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=RFE_TOP_K,
        step=0.1,
    )

    rfe.fit(X, y)

    selected = X.columns[rfe.support_].tolist()
    logger.info(f"Features selected by RFE: {len(selected)}")

    return selected


def run_feature_selection() -> None:
    logger.info("Starting feature selection pipeline (classification)")

    X, y = load_features()

    corr_selected = correlation_filter(X)
    X_corr = X[corr_selected]

    mi_selected = mutual_information_filter(X_corr, y)
    X_mi = X_corr[mi_selected]

    rfe_selected = rfe_filter(X_mi, y)

    selected_features = {
        "correlation_filter": corr_selected,
        "mutual_information": mi_selected,
        "rfe": rfe_selected,
        "final_features": rfe_selected,
    }

    with open(SELECTED_FEATURES_FILE, "w") as f:
        json.dump(selected_features, f, indent=2)

    logger.info("Feature selection completed successfully (classification)")


if __name__ == "__main__":
    run_feature_selection()