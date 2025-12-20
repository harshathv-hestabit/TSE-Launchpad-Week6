from pathlib import Path
from src.utils.logger import logTool

import pandas as pd
import numpy as np

RAW_DIR = Path("src/data/raw")
PROCESSED_DIR = Path("src/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASICS_FILE = RAW_DIR / "title.basics.tsv"
RATINGS_FILE = RAW_DIR / "title.ratings.tsv"

OUTPUT_FILE = PROCESSED_DIR / "final.csv"

logger = logTool(__name__)

def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {path}")

    logger.info(f"Loading {path.name}")
    df = pd.read_csv(
        path,
        sep="\t",
        na_values="\\N",
        low_memory=False,
        dtype={"tconst": "string"},
    )

    if "tconst" not in df.columns:
        raise ValueError(f"tconst column missing in {path.name}")

    if df["tconst"].isna().any():
        raise ValueError(f"Null tconst values found in {path.name}")

    return df

def filter_movies(basics: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filtering title.basics to movies only")

    before = len(basics)
    basics = basics[basics["titleType"] == "movie"].copy()
    after = len(basics)

    logger.info(f"Movies retained: {after} / {before}")
    return basics

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping non-informative columns: titleType, endYear")

    return df.drop(
        columns=["titleType", "endYear"],
        errors="ignore",
    )

def join_ratings(basics: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    logger.info("Joining title.basics with title.ratings")

    imdb = basics.merge(
        ratings,
        on="tconst",
        how="left",
        validate="many_to_one",
    )

    logger.info(f"Post-join row count: {len(imdb)}")
    return imdb

def handle_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missing target values (averageRating)")

    before = len(df)
    df = df[df["averageRating"].notna()].copy()
    after = len(df)

    logger.info(f"Rows dropped (missing target): {before - after}")
    return df

def handle_runtime_log_zscore(
    df: pd.DataFrame,
    z_threshold: float = 4,
) -> pd.DataFrame:
    logger.info(
        f"Handling runtimeMinutes using log-Z-score "
        f"(threshold={z_threshold})"
    )

    before = len(df)

    df["runtimeMinutes"] = pd.to_numeric(
        df["runtimeMinutes"], errors="coerce"
    )
    df = df[df["runtimeMinutes"] > 0].copy()

    log_runtime = np.log(df["runtimeMinutes"])
    z = (log_runtime - log_runtime.mean()) / log_runtime.std()

    df = df[z.abs() <= z_threshold].copy()

    after = len(df)

    logger.info(
        f"runtimeMinutes cleaned | Dropped: {before - after}"
    )
    return df

def handle_numvotes_log_zscore(
    df: pd.DataFrame,
    z_threshold: float = 4,
) -> pd.DataFrame:
    logger.info(
        f"Handling numVotes using log-Z-score "
        f"(threshold={z_threshold})"
    )

    before = len(df)

    df["numVotes"] = pd.to_numeric(
        df["numVotes"], errors="coerce"
    )
    df = df[df["numVotes"] > 0].copy()

    log_votes = np.log(df["numVotes"])
    z = (log_votes - log_votes.mean()) / log_votes.std()

    df = df[z.abs() <= z_threshold].copy()

    after = len(df)

    logger.info(
        f"numVotes cleaned | Dropped: {before - after}"
    )
    return df

def clean_start_year(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning startYear")

    df["startYear"] = pd.to_numeric(
        df["startYear"], errors="coerce"
    )

    before = len(df)
    df = df[df["startYear"].notna()].copy()
    df["startYear"] = df["startYear"].astype(int)
    after = len(df)

    logger.info(f"startYear cleaned | Dropped: {before - after}")
    return df

def run_pipeline() -> None:
    logger.info("Starting preprocessing pipeline")

    basics = load_tsv(BASICS_FILE)
    ratings = load_tsv(RATINGS_FILE)

    basics = filter_movies(basics)

    imdb = join_ratings(basics, ratings)

    imdb = drop_unused_columns(imdb)

    imdb = handle_missing_targets(imdb)

    imdb = clean_start_year(imdb)

    imdb = handle_runtime_log_zscore(imdb)

    imdb = handle_numvotes_log_zscore(imdb)

    logger.info(f"Saving processed dataset to {OUTPUT_FILE}")
    imdb.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    run_pipeline()