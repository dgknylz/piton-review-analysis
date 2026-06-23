"""Fuzzy logic based review reliability scoring."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "reliability_sample.csv"


@lru_cache(maxsize=1)
def _build_control_system() -> ctrl.ControlSystem:
    """Create the fuzzy inference system.

    Review length and age ranges use fixed, defensible caps for e-commerce reviews:
    0-1000 words and 0-3650 days. Incoming values are clipped to those ranges.
    """
    rating = ctrl.Antecedent(np.arange(0, 5.1, 0.1), "rating")
    review_length = ctrl.Antecedent(np.arange(0, 1001, 1), "review_length")
    review_age = ctrl.Antecedent(np.arange(0, 3651, 1), "review_age_days")
    reliability = ctrl.Consequent(np.arange(0, 101, 1), "reliability_score")

    rating["low"] = fuzz.trapmf(rating.universe, [0, 0, 1.5, 2.5])
    rating["medium"] = fuzz.trimf(rating.universe, [2, 3, 4])
    rating["high"] = fuzz.trapmf(rating.universe, [3.5, 4.2, 5, 5])

    review_length["short"] = fuzz.trapmf(review_length.universe, [0, 0, 20, 60])
    review_length["medium"] = fuzz.trimf(review_length.universe, [30, 120, 250])
    review_length["long"] = fuzz.trapmf(review_length.universe, [180, 400, 1000, 1000])

    review_age["new"] = fuzz.trapmf(review_age.universe, [0, 0, 30, 90])
    review_age["recent"] = fuzz.trimf(review_age.universe, [60, 180, 365])
    review_age["old"] = fuzz.trapmf(review_age.universe, [300, 730, 3650, 3650])

    reliability["low"] = fuzz.trapmf(reliability.universe, [0, 0, 25, 45])
    reliability["medium"] = fuzz.trimf(reliability.universe, [35, 55, 75])
    reliability["high"] = fuzz.trapmf(reliability.universe, [65, 80, 100, 100])

    rules = [
        ctrl.Rule(rating["high"] & review_length["long"] & review_age["new"], reliability["high"]),
        ctrl.Rule(review_length["short"] & review_age["old"], reliability["low"]),
        ctrl.Rule(rating["medium"] & review_length["medium"], reliability["medium"]),
        ctrl.Rule(rating["low"] & review_length["long"], reliability["high"]),
        ctrl.Rule(review_length["short"] & rating["medium"], reliability["low"]),
        ctrl.Rule(review_length["long"] & review_age["recent"], reliability["high"]),
        ctrl.Rule(review_length["medium"] & review_age["recent"], reliability["medium"]),
        ctrl.Rule(review_length["short"], reliability["low"]),
        ctrl.Rule(review_length["medium"] & rating["high"], reliability["medium"]),
        ctrl.Rule(review_length["medium"] & rating["low"], reliability["medium"]),
        ctrl.Rule(review_length["long"] & rating["high"], reliability["high"]),
        ctrl.Rule(review_age["old"] & review_length["long"], reliability["medium"]),
    ]
    return ctrl.ControlSystem(rules)


def _fallback_reliability(rating: float, review_length: float, review_age_days: float) -> float:
    length_component = min(max(review_length / 250, 0), 1) * 45
    recency_component = (1 - min(max(review_age_days / 3650, 0), 1)) * 25
    rating_component = min(max(abs(rating - 3) / 2, 0), 1) * 30
    return round(length_component + recency_component + rating_component, 2)


def compute_reliability_score(
    rating: float,
    review_length: float,
    review_age_days: float,
) -> float:
    """Compute a 0-100 rule-based reliability score for a review."""
    rating = float(np.clip(rating, 0, 5))
    review_length = float(np.clip(review_length, 0, 1000))
    review_age_days = float(np.clip(review_age_days, 0, 3650))

    try:
        simulation = ctrl.ControlSystemSimulation(_build_control_system())
        simulation.input["rating"] = rating
        simulation.input["review_length"] = review_length
        simulation.input["review_age_days"] = review_age_days
        simulation.compute()
        return round(float(simulation.output["reliability_score"]), 2)
    except Exception:
        return _fallback_reliability(rating, review_length, review_age_days)


def apply_reliability_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a reliability_score column."""
    scored = df.copy()
    scored["reliability_score"] = scored.apply(
        lambda row: compute_reliability_score(
            rating=float(row.get("star_rating", row.get("rating", 0))),
            review_length=float(row.get("review_length", 0)),
            review_age_days=float(row.get("review_age_days", 0)),
        ),
        axis=1,
    )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sample reliability score report.")
    parser.add_argument("--input-path", default=str(DEFAULT_CLEAN_PATH), help="Clean review CSV path.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=500, help="Rows to score for a sample report.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError("Clean review file not found. Run python -m src.data_loader first.")

    df = pd.read_csv(input_path, parse_dates=["review_date"])
    scored = apply_reliability_scores(df.head(args.rows))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Saved reliability sample to {output_path}")


if __name__ == "__main__":
    main()

