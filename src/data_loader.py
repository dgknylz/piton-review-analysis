"""Data loading and clean sampling utilities for Amazon review TSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews_us_Electronics_v1_00.tsv"
DEFAULT_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"

REQUIRED_COLUMNS = ["review_body", "star_rating", "review_date"]
OPTIONAL_COLUMNS = [
    "product_id",
    "product_parent",
    "product_title",
    "product_category",
    "helpful_votes",
    "total_votes",
    "verified_purchase",
    "review_headline",
]
READ_COLUMNS = OPTIONAL_COLUMNS + REQUIRED_COLUMNS


def load_raw_reviews(path: str | Path) -> pd.DataFrame:
    """Load the raw Amazon reviews TSV while selecting only useful columns."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {input_path}. Place the Kaggle TSV in data/raw first."
        )

    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                input_path,
                sep="\t",
                usecols=lambda column: column in READ_COLUMNS,
                on_bad_lines="skip",
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode {input_path}; last error: {last_error}",
    )


def map_star_to_sentiment(star_rating: int | float) -> str:
    """Map Amazon 1-5 star ratings into negative, neutral, and positive labels."""
    rating = int(star_rating)
    if rating in (1, 2):
        return "negative"
    if rating == 3:
        return "neutral"
    if rating in (4, 5):
        return "positive"
    raise ValueError(f"star_rating must be between 1 and 5, got {star_rating!r}")


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")


def _stratified_sample(
    df: pd.DataFrame, sample_size: int, random_state: int
) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    try:
        from sklearn.model_selection import train_test_split

        sampled, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=random_state,
            stratify=df["sentiment"],
        )
        return sampled.reset_index(drop=True)
    except ValueError:
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def create_clean_sample(
    input_path: str | Path = DEFAULT_RAW_PATH,
    output_path: str | Path = DEFAULT_CLEAN_PATH,
    sample_size: int = 30_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a clean, reproducible sample from the local raw Kaggle TSV."""
    df = load_raw_reviews(input_path)
    _ensure_required_columns(df)

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df.dropna(subset=["review_body", "star_rating", "review_date"]).copy()
    df["review_headline"] = df["review_headline"].fillna("")
    df["review_body"] = df["review_body"].fillna("")
    df["review_text"] = (
        df["review_headline"].astype(str).str.strip()
        + " "
        + df["review_body"].astype(str).str.strip()
    ).str.strip()

    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
    df = df.dropna(subset=["star_rating"])
    df["star_rating"] = df["star_rating"].astype(int)
    df = df[df["star_rating"].between(1, 5)].copy()
    df["sentiment"] = df["star_rating"].map(map_star_to_sentiment)

    df["review_length"] = df["review_text"].str.split().str.len().fillna(0).astype(int)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["review_date"]).copy()
    today = pd.Timestamp.today().normalize()
    df["review_age_days"] = (today - df["review_date"].dt.normalize()).dt.days.clip(lower=0)

    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0)
    df["total_votes"] = pd.to_numeric(df["total_votes"], errors="coerce").fillna(0)
    df["helpful_ratio"] = np.where(
        df["total_votes"] > 0,
        df["helpful_votes"] / df["total_votes"],
        0.0,
    )

    final_columns = [
        "product_id",
        "product_parent",
        "product_title",
        "product_category",
        "star_rating",
        "helpful_votes",
        "total_votes",
        "helpful_ratio",
        "verified_purchase",
        "review_headline",
        "review_body",
        "review_text",
        "sentiment",
        "review_length",
        "review_date",
        "review_age_days",
    ]
    df = df[[column for column in final_columns if column in df.columns]].copy()
    sampled = _stratified_sample(df, sample_size=sample_size, random_state=random_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean sampled review dataset.")
    parser.add_argument("--input-path", default=str(DEFAULT_RAW_PATH), help="Local raw TSV path.")
    parser.add_argument("--output-path", default=str(DEFAULT_CLEAN_PATH), help="Clean CSV output path.")
    parser.add_argument("--sample-size", type=int, default=30_000, help="Sample size for modeling.")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducible random seed.")
    args = parser.parse_args()

    df = create_clean_sample(
        input_path=args.input_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print(f"Saved clean sample with shape {df.shape} to {args.output_path}")


if __name__ == "__main__":
    main()
