"""Binary-status nutrient models, compared on the locked participant split."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .experiment import split_people, save_split, food_features
from .metrics import metrics


def diabetes_flags(cohorts):
    cohorts = np.asarray(cohorts)
    if not np.isin(cohorts, ["healthy", "prediabetes", "type2_diabetes"]).all():
        raise ValueError("Unknown CGMacros cohort")
    return (cohorts == "type2_diabetes").astype("float32")


def validate_flag(value):
    if value not in (0, 1):
        raise ValueError("Diabetes flag must be 0 (non-diabetic) or 1 (type 2 diabetic)")
    return int(value)


def candidates(seed):
    return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
            "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8,
                                               random_state=seed, n_jobs=4)}


def train_status(prepared="data/prepared", output="artifacts/status", seed=42):
    started = time.perf_counter()
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(prepared / "windows.npz")
    meals = pd.read_csv(prepared / "meals.csv")
    h, y, macro = [arrays[k] for k in ["history", "target", "macros"]]
    flags = diabetes_flags(meals.cohort)
    base = h[:, -1:]
    delta = y - base
    plain = food_features(macro, h)
    flagged = np.column_stack([plain, flags])
    split = split_people(meals.participant.to_numpy(), seed, meals.cohort.to_numpy())
    save_split(meals, split, output)
    idx = {k: np.flatnonzero(meals.participant.isin(v)) for k, v in split.items()}
    tr, va, te = [idx[k] for k in ["train", "validation", "test"]]
    predictions, selections, validation = {}, {}, {}
    for label, features in [("pooled", plain), ("binary_flag", flagged)]:
        fitted = candidates(seed)
        scores = {}
        for name, model in fitted.items():
            model.fit(features[tr], delta[tr])
            scores[name] = metrics(y[va], model.predict(features[va]) + base[va])
        selected = min(scores, key=lambda k: scores[k]["peak_mae_mg_dl"])
        validation[label] = scores
        selections[label] = selected
        predictions[label] = fitted[selected].predict(features) + base
        joblib.dump(fitted[selected], output / ("food_model.joblib" if label == "binary_flag" else "pooled_model.joblib"))
    separate = np.empty_like(y)
    group_counts = {}
    for flag, label in [(0, "non_diabetic"), (1, "diabetic")]:
        gtr, gva, gte = [indices[flags[indices] == flag] for indices in [tr, va, te]]
        fitted = candidates(seed)
        scores = {}
        for name, model in fitted.items():
            model.fit(plain[gtr], delta[gtr])
            scores[name] = metrics(y[gva], model.predict(plain[gva]) + base[gva])
        selected = min(scores, key=lambda k: scores[k]["peak_mae_mg_dl"])
        mask = np.flatnonzero(flags == flag)
        separate[mask] = fitted[selected].predict(plain[mask]) + base[mask]
        joblib.dump(fitted[selected], output / f"{label}_model.joblib")
        validation[label], selections[label] = scores, selected
        group_counts[label] = {key: {"participants": int(meals.iloc[indices].participant.nunique()),
            "meals": len(indices)} for key, indices in zip(["train", "validation", "test"], [gtr, gva, gte])}
    predictions["separate_models"] = separate
    test_metrics = {name: metrics(y[te], pred[te]) for name, pred in predictions.items()}
    subgroup_metrics = {}
    for cohort in ["healthy", "prediabetes", "type2_diabetes"]:
        subset = te[meals.iloc[te].cohort.to_numpy() == cohort]
        subgroup_metrics[cohort] = {name: metrics(y[subset], pred[subset]) for name, pred in predictions.items()}
    report = {"version": "binary diabetes flag comparison", "seed": seed, "split": split,
        "split_meals": {k: len(v) for k, v in idx.items()}, "group_split_counts": group_counts,
        "flag_definition": {"0": "healthy or prediabetes", "1": "type2_diabetes"},
        "feature_order": ["Carbs", "Protein", "Fat", "Fiber", "starting_glucose_mg_dl", "diabetes_flag"],
        "selected_model": "binary_flag", "selected_estimators": selections,
        "validation": validation, "test": test_metrics, "cohort_test": subgroup_metrics,
        "training_feature_min": flagged[tr].min(0).tolist(), "training_feature_max": flagged[tr].max(0).tolist(),
        "training_evaluation_seconds": time.perf_counter() - started}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    frame = meals.iloc[te].copy()
    frame["diabetes_flag"] = flags[te]
    frame["actual_peak"] = y[te].max(1)
    for name, pred in predictions.items():
        frame[name + "_peak"] = pred[te].max(1)
    frame.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "curves.npz", target=y[te], history=h[te], **{k: v[te] for k, v in predictions.items()})
    cohort_rows = [{"cohort": cohort, "model": model, **score} for cohort, models in subgroup_metrics.items() for model, score in models.items()]
    pd.DataFrame(cohort_rows).to_csv(output / "cohort_metrics.csv", index=False)
    return report
