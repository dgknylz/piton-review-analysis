"""Single-review prediction helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.preprocessing import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_sentiment_model.joblib"


def predict_sentiment(review_text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, object]:
    """Predict sentiment for a single raw review text."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Please run the training pipeline first.")

    model = joblib.load(model_path)
    processed = preprocess_text(review_text)
    prediction = model.predict([processed])[0]
    result: dict[str, object] = {"sentiment": prediction, "processed_text": processed}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([processed])[0]
        classes = list(model.classes_)
        result["confidence"] = float(probabilities.max())
        result["probabilities"] = {
            label: float(probabilities[index]) for index, label in enumerate(classes)
        }
    else:
        result["confidence"] = None
        result["probabilities"] = {}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for one review.")
    parser.add_argument("review_text", help="Review text to classify.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Saved model path.")
    args = parser.parse_args()
    print(predict_sentiment(args.review_text, args.model_path))


if __name__ == "__main__":
    main()

