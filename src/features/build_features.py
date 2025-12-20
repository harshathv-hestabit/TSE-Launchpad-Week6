from pathlib import Path
from src.utils.logger import logTool

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import json

DATA_DIR = Path("src/data/processed")
FEATURE_DIR = Path("src/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "final.csv"

X_TRAIN_FILE = DATA_DIR / "X_train.csv"
X_TEST_FILE = DATA_DIR / "X_test.csv"
Y_TRAIN_FILE = DATA_DIR / "y_train.csv"
Y_TEST_FILE = DATA_DIR / "y_test.csv"

FEATURE_LIST_FILE = FEATURE_DIR / "feature_list.json"

TARGET_COL_RAW = "averageRating"
TARGET_COL = "rating_class"

NUMERIC_COLS = [
    "runtimeMinutes",
    "numVotes",
]

CATEGORICAL_COLS = [
    "genres",
]

RANDOM_STATE = 42
TEST_SIZE = 0.3

logger = logTool(__name__)

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    logger.info(f"Loading processed dataset: {path}")
    return pd.read_csv(path)


def encode_ordinal_target(ratings: pd.Series) -> pd.Series:
    bins = [-np.inf, 4.5, 6.0, 7.0, 8.0, np.inf]
    labels = [0, 1, 2, 3, 4]

    return pd.cut(ratings, bins=bins, labels=labels).astype(int)


def process_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing numeric features")

    num_df = df[NUMERIC_COLS].copy()

    for col in NUMERIC_COLS:
        num_df[col] = num_df[col].fillna(num_df[col].median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)

    return pd.DataFrame(
        scaled,
        columns=num_df.columns,
        index=df.index,
    )


def process_genres(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing categorical feature: genres")

    genres = df["genres"].fillna("Unknown")
    genres_split = genres.str.split(",")

    return genres_split.str.join("|").str.get_dummies()


def run_feature_pipeline() -> None:
    logger.info("Building Features")

    df = load_dataset(INPUT_FILE)

    y = encode_ordinal_target(df[TARGET_COL_RAW])

    logger.info("Target class distribution:")
    logger.info(y.value_counts().sort_index())

    X_num = process_numeric_features(df)
    X_cat = process_genres(df)
    X = pd.concat([X_num, X_cat], axis=1)

    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Final Columns {X.columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train.to_csv(X_TRAIN_FILE, index=False)
    X_test.to_csv(X_TEST_FILE, index=False)
    y_train.to_csv(Y_TRAIN_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)

    feature_list = {
        "numeric_features": list(X_num.columns),
        "categorical_features": list(X_cat.columns),
        "all_features": list(X.columns),
        "target": TARGET_COL,
        "ordinal_classes": {
            "0": "Very Poor",
            "1": "Poor",
            "2": "Average",
            "3": "Good",
            "4": "Excellent",
        },
    }

    with open(FEATURE_LIST_FILE, "w") as f:
        json.dump(feature_list, f, indent=2)

    logger.info("Feature engineering completed successfully (classification only)")


if __name__ == "__main__":
    run_feature_pipeline()