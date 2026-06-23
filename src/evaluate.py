"""Model evaluation, reporting, and error analysis helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


LABEL_ORDER = ["negative", "neutral", "positive"]


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    labels = labels or LABEL_ORDER
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model(
    model,
    x_test,
    y_test,
    model_name: str,
    figures_dir: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate a fitted classifier pipeline and save its confusion matrix."""
    y_pred = model.predict(x_test)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
        ),
        "classification_report_dict": classification_report(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
        "y_pred": y_pred,
    }

    if figures_dir is not None:
        output_path = Path(figures_dir) / f"confusion_matrix_{_safe_filename(model_name)}.png"
        plot_confusion_matrix(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            title=f"Confusion Matrix - {model_name}",
            output_path=output_path,
        )

    return metrics


def save_model_comparison(metrics_list: list[dict[str, object]], output_path: str | Path) -> pd.DataFrame:
    """Save accuracy, macro F1, and weighted F1 for model comparison."""
    rows = [
        {
            "model": metrics["model"],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        }
        for metrics in metrics_list
    ]
    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def save_classification_report(report_text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def prediction_confidence(model, x_values) -> list[float | None]:
    """Return max class probability when the model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        return [None for _ in range(len(x_values))]
    probabilities = model.predict_proba(x_values)
    return probabilities.max(axis=1).tolist()


def make_error_comment(row: pd.Series) -> str:
    """Create a short human-readable comment for a misclassified example."""
    if row.get("review_length", 0) <= 5:
        return "Very short review gives the model limited context."
    if row.get("true_label") == "neutral" or row.get("predicted_label") == "neutral":
        return "Neutral reviews often overlap with positive or negative wording."
    if row.get("star_rating") in (1, 2) and row.get("predicted_label") == "positive":
        return "Low rating may conflict with positive words in the review text."
    if row.get("star_rating") in (4, 5) and row.get("predicted_label") == "negative":
        return "High rating may conflict with negative details in the text."
    return "Likely mixed, ambiguous, or context-dependent sentiment."


def create_error_analysis_table(
    model,
    x_test,
    y_test: pd.Series,
    metadata: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a table of high-confidence misclassifications for qualitative review."""
    from src.fuzzy_system import compute_reliability_score

    y_pred = pd.Series(model.predict(x_test), index=y_test.index, name="predicted_label")
    confidences = pd.Series(prediction_confidence(model, x_test), index=y_test.index, name="model_confidence")

    rows = metadata.loc[y_test.index].copy()
    rows["true_label"] = y_test
    rows["predicted_label"] = y_pred
    rows["model_confidence"] = confidences
    errors = rows[rows["true_label"] != rows["predicted_label"]].copy()

    if errors.empty:
        return pd.DataFrame(
            columns=[
                "review_text",
                "true_label",
                "predicted_label",
                "model_confidence",
                "star_rating",
                "review_length",
                "review_age_days",
                "reliability_score",
                "short_error_comment",
            ]
        )

    errors["reliability_score"] = errors.apply(
        lambda row: compute_reliability_score(
            rating=float(row.get("star_rating", 0)),
            review_length=float(row.get("review_length", 0)),
            review_age_days=float(row.get("review_age_days", 0)),
        ),
        axis=1,
    )
    errors["short_error_comment"] = errors.apply(make_error_comment, axis=1)

    columns = [
        "review_text",
        "true_label",
        "predicted_label",
        "model_confidence",
        "star_rating",
        "review_length",
        "review_age_days",
        "reliability_score",
        "short_error_comment",
    ]
    return (
        errors[columns]
        .sort_values("model_confidence", ascending=False, na_position="last")
        .head(top_n)
    )

