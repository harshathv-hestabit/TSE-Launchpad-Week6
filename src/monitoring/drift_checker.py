import json
import time
import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from scipy.stats import ks_2samp
from datetime import datetime

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
VOLUME_DIR = Path(os.getenv("LOG_PATH"))
LOG_FILE = VOLUME_DIR / "prediction_logs.csv"

NUMERIC_FEATURES = ["runtimeMinutes", "numVotes"]

GENRE_COLUMNS = [
    "Action", "Adult", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir",
    "Game-Show", "History", "Horror", "Music", "Musical", "Mystery",
    "News", "Reality-TV", "Romance", "Sci-Fi", "Sport", "Talk-Show",
    "Thriller", "Unknown", "War", "Western"
]

DRIFT_THRESHOLD_PVALUE = 0.05

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def load_reference_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    return X_train

def load_prediction_data():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"prediction_logs.csv not found at {LOG_FILE}")

    logs = pd.read_csv(LOG_FILE)

    data = {
        "runtimeMinutes": logs["runtimeMinutes"].astype(float),
        "numVotes": logs["numVotes"].astype(float),
    }

    for genre in GENRE_COLUMNS:
        data[genre] = logs["genres"].str.contains(genre).astype(int)

    return pd.DataFrame(data)

def ks_drift_test(ref: pd.Series, prod: pd.Series):
    stat, p_value = ks_2samp(ref, prod)
    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < DRIFT_THRESHOLD_PVALUE)
    }

def detect_drift(X_ref, X_prod):
    drift_report = {}

    for feature in NUMERIC_FEATURES:
        drift_report[feature] = ks_drift_test(
            X_ref[feature].dropna(),
            X_prod[feature].dropna()
        )

    for genre in GENRE_COLUMNS:
        ref_rate = X_ref[genre].mean()
        prod_rate = X_prod[genre].mean()

        drift_report[genre] = {
            "reference_rate": float(ref_rate),
            "production_rate": float(prod_rate),
            "absolute_change": float(abs(ref_rate - prod_rate)),
            "drift_detected": bool(abs(ref_rate - prod_rate) > 0.05)
        }

    return drift_report

def run_drift():
    X_ref = load_reference_data()
    X_prod = load_prediction_data()
    drift_results = detect_drift(X_ref, X_prod)

    output = {
        "timestamp": datetime.now().isoformat(),
        "drift_threshold_pvalue": DRIFT_THRESHOLD_PVALUE,
        "drift_report": drift_results
    }

    with open(VOLUME_DIR / "drift_report.json", "w") as f:
        json.dump(output, f, indent=4, cls=NpEncoder)

    print("Drift analysis completed")
    print(f"Report saved to {VOLUME_DIR / 'drift_report.json'}")

if __name__ == "__main__":
    while True:
        run_drift()
        time.sleep(3600)