"""Business-oriented complaint keyword and phrase extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.data_loader import map_star_to_sentiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "top_complaints.csv"


def _ensure_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "sentiment" not in result.columns and "star_rating" in result.columns:
        result["sentiment"] = result["star_rating"].map(map_star_to_sentiment)
    return result


def get_top_complaints(
    df: pd.DataFrame,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top complaint terms or bigrams for negative reviews."""
    filtered = _ensure_sentiment(df)
    filtered = filtered[filtered["sentiment"] == "negative"].copy()

    if category and "product_category" in filtered.columns:
        filtered = filtered[filtered["product_category"].astype(str).str.lower() == category.lower()]

    if "review_date" in filtered.columns:
        filtered["review_date"] = pd.to_datetime(filtered["review_date"], errors="coerce")
        if start_date:
            filtered = filtered[filtered["review_date"] >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[filtered["review_date"] <= pd.to_datetime(end_date)]

    if filtered.empty or "review_text" not in filtered.columns:
        return pd.DataFrame(columns=["complaint_term", "count"])

    texts = filtered["review_text"].fillna("").astype(str)
    texts = texts[texts.str.strip().ne("")]
    if texts.empty:
        return pd.DataFrame(columns=["complaint_term", "count"])

    min_df = 1 if len(texts) < 3 else 2
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    summary = (
        pd.DataFrame({"complaint_term": terms, "count": counts})
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top complaint terms.")
    parser.add_argument("--input-path", default=str(DEFAULT_CLEAN_PATH), help="Clean review CSV path.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path.")
    parser.add_argument("--category", default=None, help="Optional product category filter.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end date.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of terms to return.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError("Clean review file not found. Run python -m src.data_loader first.")

    df = pd.read_csv(input_path, parse_dates=["review_date"])
    summary = get_top_complaints(
        df,
        category=args.category,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved complaint summary to {output_path}")


if __name__ == "__main__":
    main()

