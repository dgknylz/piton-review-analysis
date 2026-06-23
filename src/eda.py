"""Reusable exploratory data analysis helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and ratios for every column."""
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_ratio": df.isna().mean(),
        }
    )
    return summary.sort_values("missing_ratio", ascending=False)


def get_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric project columns."""
    numeric_columns = [
        column
        for column in [
            "star_rating",
            "helpful_votes",
            "total_votes",
            "helpful_ratio",
            "review_length",
            "review_age_days",
        ]
        if column in df.columns
    ]
    return df[numeric_columns].describe().T if numeric_columns else pd.DataFrame()


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Create the required EDA plots and save them under outputs/figures."""
    output_dir = Path(output_dir)
    sns.set_theme(style="whitegrid")
    saved_paths: dict[str, Path] = {}

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_ratio.index, y=missing_ratio.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing ratio")
    plt.title("Missing Value Ratios")
    saved_paths["missing_values"] = output_dir / "missing_values.png"
    _save_current_figure(saved_paths["missing_values"])

    if "star_rating" in df.columns:
        plt.figure(figsize=(7, 4))
        order = sorted(df["star_rating"].dropna().unique())
        sns.countplot(data=df, x="star_rating", order=order)
        plt.title("Star Rating Distribution")
        plt.xlabel("Star rating")
        plt.ylabel("Review count")
        saved_paths["rating_distribution"] = output_dir / "rating_distribution.png"
        _save_current_figure(saved_paths["rating_distribution"])

    if "sentiment" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x="sentiment", order=SENTIMENT_ORDER)
        plt.title("Sentiment Class Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Review count")
        saved_paths["sentiment_distribution"] = output_dir / "sentiment_distribution.png"
        _save_current_figure(saved_paths["sentiment_distribution"])

    if "review_length" in df.columns:
        length_cap = df["review_length"].quantile(0.99)
        plt.figure(figsize=(8, 4))
        sns.histplot(df["review_length"].clip(upper=length_cap), bins=40, kde=True)
        plt.title("Review Length Distribution")
        plt.xlabel("Review length (word count, capped at p99)")
        saved_paths["review_length_distribution"] = output_dir / "review_length_distribution.png"
        _save_current_figure(saved_paths["review_length_distribution"])

    if "review_age_days" in df.columns:
        age_cap = df["review_age_days"].quantile(0.99)
        plt.figure(figsize=(8, 4))
        sns.histplot(df["review_age_days"].clip(upper=age_cap), bins=40, kde=True)
        plt.title("Review Age Distribution")
        plt.xlabel("Review age in days (capped at p99)")
        saved_paths["review_age_distribution"] = output_dir / "review_age_distribution.png"
        _save_current_figure(saved_paths["review_age_distribution"])

    return saved_paths


def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return sentiment counts and percentages for class imbalance discussion."""
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "count": counts,
            "ratio": counts / max(counts.sum(), 1),
        }
    )

