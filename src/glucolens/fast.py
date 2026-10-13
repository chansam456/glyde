"""CPU-first nutrient model; image embeddings are not required."""
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
from .experiment import split_people, food_features, save_split
from .metrics import metrics


def train_fast(prepared, output, seed=42):
    started = time.perf_counter()
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(prepared / "windows.npz")
    meals = pd.read_csv(prepared / "meals.csv")
    h, y, macro = [arrays[x] for x in ["history", "target", "macros"]]
    x = food_features(macro, h)
    baseline = h[:, -1:]
    target = y - baseline
    split = split_people(meals.participant.to_numpy(), seed, meals.cohort.to_numpy())
    save_split(meals, split, output)
    idx = {k: np.flatnonzero(meals.participant.isin(v)) for k, v in split.items()}
    tr, va, te = [idx[k] for k in ["train", "validation", "test"]]
    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, random_state=seed, n_jobs=4),
    }
    predictions, validation = {}, {}
    for name, model in candidates.items():
        model.fit(x[tr], target[tr])
        predictions[name] = model.predict(x) + baseline
        validation[name] = metrics(y[va], predictions[name][va])
    selected = min(candidates, key=lambda k: validation[k]["peak_mae_mg_dl"])
    predictions["persistence"] = np.repeat(baseline, 24, axis=1)
    predictions["population_response"] = target[tr].mean(0)[None, :] + baseline
    no_food = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(x[tr, -1:], target[tr])
    predictions["glucose_only_ridge"] = no_food.predict(x[:, -1:]) + baseline
    report = {"version": "v1 nutrient model", "seed": seed, "split": split,
        "split_meals": {k: len(v) for k, v in idx.items()}, "selected_model": selected,
        "validation": validation, "test": {k: metrics(y[te], v[te]) for k, v in predictions.items()},
        "training_evaluation_seconds": time.perf_counter() - started,
        "feature_order": ["Carbs", "Protein", "Fat", "Fiber", "starting_glucose_mg_dl"],
        "training_feature_min": x[tr].min(0).tolist(), "training_feature_max": x[tr].max(0).tolist()}
    joblib.dump(candidates[selected], output / "food_model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "curves.npz", target=y[te], history=h[te],
                        **{k: v[te] for k, v in predictions.items()})
    frame = meals.iloc[te].copy()
    frame["actual_peak"] = y[te].max(1)
    for name, prediction in predictions.items():
        frame[name + "_peak"] = prediction[te].max(1)
    frame.to_csv(output / "predictions.csv", index=False)
    cohort_rows, person_rows = [], []
    for cohort in sorted(meals.cohort.unique()):
        subset = te[meals.iloc[te].cohort.to_numpy() == cohort]
        for name, pred in predictions.items():
            cohort_rows.append({"cohort": cohort, "model": name, **metrics(y[subset], pred[subset])})
    for participant in split["test"]:
        subset = te[meals.iloc[te].participant.to_numpy() == participant]
        for name, pred in predictions.items():
            person_rows.append({"participant": participant, "model": name, **metrics(y[subset], pred[subset])})
    pd.DataFrame(cohort_rows).to_csv(output / "cohort_metrics.csv", index=False)
    pd.DataFrame(person_rows).to_csv(output / "participant_metrics.csv", index=False)
    return report
