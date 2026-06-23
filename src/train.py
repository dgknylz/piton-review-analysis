"""Train, compare, optimize, and save sentiment classification models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import DEFAULT_CLEAN_PATH, DEFAULT_RAW_PATH, create_clean_sample
from src.evaluate import (
    LABEL_ORDER,
    create_error_analysis_table,
    evaluate_model,
    plot_confusion_matrix,
    save_classification_report,
    save_model_comparison,
)
from src.preprocessing import ensure_nltk_resources, preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
BEST_MODEL_PATH = MODELS_DIR / "best_sentiment_model.joblib"


def _ensure_output_dirs() -> None:
    for path in (FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_or_create_clean_data(
    clean_path: str | Path = DEFAULT_CLEAN_PATH,
    raw_path: str | Path = DEFAULT_RAW_PATH,
    sample_size: int = 30_000,
) -> pd.DataFrame:
    """Load processed data if available, otherwise create it from the local raw TSV."""
    clean_path = Path(clean_path)
    raw_path = Path(raw_path)
    if clean_path.exists():
        return pd.read_csv(clean_path, parse_dates=["review_date"])
    if raw_path.exists():
        return create_clean_sample(raw_path, clean_path, sample_size=sample_size)
    raise FileNotFoundError(
        "No clean or raw dataset found. Place amazon_reviews_us_Electronics_v1_00.tsv "
        "under data/raw or run the data loader after adding the file."
    )


def prepare_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a processed_text column used by TF-IDF models."""
    prepared = df.copy()
    prepared["review_text"] = prepared["review_text"].fillna("").astype(str)
    prepared["processed_text"] = prepared["review_text"].map(preprocess_text)
    prepared = prepared[prepared["processed_text"].str.strip().ne("")]
    return prepared


def build_logistic_pipeline(max_features: int = 20_000, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight=class_weight),
            ),
        ]
    )


def build_random_forest_pipeline(max_features: int = 20_000) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_grid_search(x_train, y_train) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    param_grid = {
        "tfidf__max_features": [10_000, 20_000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1, 3, 10],
        "clf__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate sentiment models.")
    parser.add_argument("--clean-path", default=str(DEFAULT_CLEAN_PATH), help="Processed clean CSV path.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH), help="Local raw TSV path.")
    parser.add_argument("--sample-size", type=int, default=30_000, help="Sample size if clean CSV is absent.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--max-features", type=int, default=20_000, help="TF-IDF max_features for baselines.")
    parser.add_argument("--skip-random-forest", action="store_true", help="Skip the slower Random Forest baseline.")
    parser.add_argument("--skip-grid-search", action="store_true", help="Skip GridSearchCV optimization.")
    parser.add_argument("--download-nltk", action="store_true", help="Download optional NLTK corpora before preprocessing.")
    args = parser.parse_args()

    _ensure_output_dirs()
    ensure_nltk_resources(download=args.download_nltk)

    df = load_or_create_clean_data(args.clean_path, args.raw_path, sample_size=args.sample_size)
    df = prepare_text_column(df)

    x = df["processed_text"]
    y = df["sentiment"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    metrics = []

    logistic = build_logistic_pipeline(max_features=args.max_features, class_weight="balanced")
    logistic.fit(x_train, y_train)
    logistic_metrics = evaluate_model(logistic, x_test, y_test, "Logistic Regression", FIGURES_DIR)
    metrics.append(logistic_metrics)
    save_classification_report(
        logistic_metrics["classification_report"],
        REPORTS_DIR / "classification_report_logistic_regression.txt",
    )

    if not args.skip_random_forest:
        random_forest = build_random_forest_pipeline(max_features=args.max_features)
        random_forest.fit(x_train, y_train)
        rf_metrics = evaluate_model(random_forest, x_test, y_test, "Random Forest", FIGURES_DIR)
        metrics.append(rf_metrics)
        save_classification_report(
            rf_metrics["classification_report"],
            REPORTS_DIR / "classification_report_random_forest.txt",
        )

    optimized_metrics = None
    optimized_model = None
    if not args.skip_grid_search:
        search = run_grid_search(x_train, y_train)
        optimized_model = search.best_estimator_
        optimized_metrics = evaluate_model(optimized_model, x_test, y_test, "Optimized Logistic Regression", None)
        pd.DataFrame(search.cv_results_).to_csv(REPORTS_DIR / "grid_search_results.csv", index=False)
        pd.DataFrame([search.best_params_]).to_csv(REPORTS_DIR / "best_grid_params.csv", index=False)
        save_classification_report(
            optimized_metrics["classification_report"],
            REPORTS_DIR / "classification_report_optimized_logistic_regression.txt",
        )
        before_after = pd.DataFrame(
            [
                {
                    "model": "Baseline Logistic Regression",
                    "accuracy": logistic_metrics["accuracy"],
                    "f1_macro": logistic_metrics["f1_macro"],
                    "f1_weighted": logistic_metrics["f1_weighted"],
                },
                {
                    "model": "Optimized Logistic Regression",
                    "accuracy": optimized_metrics["accuracy"],
                    "f1_macro": optimized_metrics["f1_macro"],
                    "f1_weighted": optimized_metrics["f1_weighted"],
                },
            ]
        )
        before_after.to_csv(REPORTS_DIR / "optimized_model_metrics.csv", index=False)
        metrics.append(optimized_metrics)

    comparison = save_model_comparison(metrics, REPORTS_DIR / "model_comparison.csv")
    best_name = comparison.iloc[0]["model"]
    best_metric = next(metric for metric in metrics if metric["model"] == best_name)
    best_model = {
        "Logistic Regression": logistic,
        "Random Forest": locals().get("random_forest"),
        "Optimized Logistic Regression": optimized_model,
    }[best_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    (MODELS_DIR / "label_mapping.json").write_text(
        json.dumps({"negative": 0, "neutral": 1, "positive": 2}, indent=2),
        encoding="utf-8",
    )
    plot_confusion_matrix(
        y_test,
        best_metric["y_pred"],
        labels=LABEL_ORDER,
        title=f"Confusion Matrix - Best Model ({best_name})",
        output_path=FIGURES_DIR / "confusion_matrix_best_model.png",
    )

    metadata = df.loc[y_test.index]
    error_table = create_error_analysis_table(best_model, x_test, y_test, metadata, top_n=5)
    error_table.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)

    print(f"Saved model comparison to {REPORTS_DIR / 'model_comparison.csv'}")
    print(f"Saved best model ({best_name}) to {BEST_MODEL_PATH}")
    print(f"Saved error analysis to {REPORTS_DIR / 'error_analysis.csv'}")


if __name__ == "__main__":
    main()

