"""
Run this script to inspect the saved model files in the models/ folder.

    python inspect_models.py

No arguments needed — it finds the models folder automatically.

Note: this helper targets older filenames (linear_baseline_model.pkl /
random_forest_candidate.pkl). The current Day 4 pipeline saves
day4_linear_regression.pkl and day4_random_forest.pkl.
Edit the paths below or use the Day 4 notebook reload cell instead.
"""

import pickle
from pathlib import Path

models_dir = Path(__file__).parent / "models"


def load_pkl(path):
    """Try to load a pkl file. Returns (object, None) on success or (None, error_message) on failure."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


# ── Linear baseline ───────────────────────────────────────────────────────────
print("=" * 50)
print("LINEAR BASELINE MODEL")
print("=" * 50)

linear_model, error = load_pkl(models_dir / "day4_linear_regression.pkl")

if error:
    print("Could not load day4_linear_regression.pkl.")
    print("Reason:", error)
    print("Fix: re-run cell [12] in day4_model_training.ipynb to regenerate it.")
else:
    print("Model type  :", linear_model["model_type"])
    print("Target      :", linear_model["target"])
    print("Features    :", linear_model["feature_columns"])
    print(f"Intercept   : {linear_model['intercept']:.4f}")
    print("Coefficients:")
    for feature, coef in linear_model["coefficients"].items():
        print(f"  {feature}: {coef:.4f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("RANDOM FOREST MODEL")
print("=" * 50)

rf_model, error = load_pkl(models_dir / "day4_random_forest.pkl")

if error:
    print("Could not load day4_random_forest.pkl.")
    print("Reason:", error)
    print("Fix: re-run cell [15] in day4_model_training.ipynb to regenerate it.")
else:
    print("Model type    :", type(rf_model).__name__)
    print("No. of trees  :", rf_model.n_estimators)
    print("Max depth     :", rf_model.max_depth)

    # Feature importances — needs the feature names from the linear model dict
    if linear_model is not None:
        print("Feature importances (higher = relied on more):")
        for feature, importance in sorted(
            zip(linear_model["feature_columns"], rf_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {feature}: {importance:.4f}")
    else:
        print("Feature importances (raw):", rf_model.feature_importances_)
