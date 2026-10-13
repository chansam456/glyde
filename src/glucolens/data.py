"""Align meal-start images, premeal history and future Dexcom targets."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

MACROS = ["Carbs", "Protein", "Fat", "Fiber"]


def sample_window(series, timestamps, max_gap=15):
    """Interpolate only short gaps and never extrapolate beyond observed support."""
    series = series.dropna().sort_index()
    if len(series) < 2:
        return None
    source = series.index.as_unit("ns").asi8
    query = pd.DatetimeIndex(timestamps).as_unit("ns").asi8
    right = np.searchsorted(source, query, side="left")
    if (query < source[0]).any() or (query > source[-1]).any():
        return None
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    exact = source[right] == query
    if (((source[right] - source[left]) / 60e9 > max_gap) & ~exact).any():
        return None
    return np.interp(query, source, series.to_numpy(dtype=float)).astype("float32")


def prepare(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    histories, targets, nutrients, rows = [], [], [], []
    excluded = Counter()
    bio_file = next(root.rglob("bio.csv"), None)
    if bio_file is None:
        raise ValueError("bio.csv is required for HbA1c-stratified participant splits")
    bio = pd.read_csv(bio_file)
    bio.columns = bio.columns.str.strip()
    a1c = dict(zip(bio.subject.astype(int), pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")))
    for file in sorted(root.rglob("CGMacros-*.csv")):
        participant = int(file.stem.split("-")[-1])
        value = a1c.get(participant, np.nan)
        if not np.isfinite(value):
            raise ValueError(f"Missing baseline HbA1c for participant {participant}")
        cohort = "healthy" if value < 5.7 else ("prediabetes" if value <= 6.4 else "type2_diabetes")
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df["Timestamp"] = pd.to_datetime(df.Timestamp)
        df = df.sort_values("Timestamp").drop_duplicates("Timestamp").set_index("Timestamp")
        glucose = pd.to_numeric(df["Dexcom GL"], errors="coerce")
        glucose = glucose.where(glucose.between(40, 400))
        meals = df[df["Meal Type"].notna()]
        for t, meal in meals.iterrows():
            future_times = t + pd.to_timedelta(np.arange(5, 121, 5), unit="min")
            history_times = t + pd.to_timedelta(np.arange(-60, 0, 5), unit="min")
            if ((meals.index > t) & (meals.index <= future_times[-1])).any():
                excluded["overlapping_meal"] += 1
                continue
            # Historical interpolation cannot see observations at/after meal onset.
            history = sample_window(glucose[glucose.index < t], history_times)
            target = sample_window(glucose, future_times)
            if history is None or target is None:
                excluded["incomplete_cgm"] += 1
                continue
            image = file.parent / str(meal.get("Image path", "")).strip().replace("\\", "/")
            if not image.is_file():
                excluded["missing_start_photo"] += 1
                continue
            macro = pd.to_numeric(meal[MACROS], errors="coerce").to_numpy(dtype="float32")
            if not np.isfinite(macro).all():
                excluded["missing_macros"] += 1
                continue
            # Broad plausibility bounds, fixed before fitting. Do not guess corrections.
            if (macro < 0).any() or (macro > np.array([300, 300, 200, 100])).any():
                excluded["implausible_macros"] += 1
                continue
            rows.append({"participant": participant, "timestamp": t.isoformat(),
                         "cohort": cohort,
                         "available_at": future_times[-1].isoformat(),
                         "image": str(image.resolve()), "meal_type": str(meal["Meal Type"]),
                         **dict(zip(MACROS, macro.tolist()))})
            histories.append(history)
            targets.append(target)
            nutrients.append(macro)
    if not rows:
        raise ValueError("No eligible meals; verify the CGMacros archive and schema")
    np.savez_compressed(output / "windows.npz", history=np.array(histories),
                        target=np.array(targets), macros=np.array(nutrients))
    pd.DataFrame(rows).to_csv(output / "meals.csv", index=False)
    audit = {"eligible_meals": len(rows), "participants": len({r['participant'] for r in rows}),
             "cohort_participants": pd.DataFrame(rows).drop_duplicates("participant").cohort.value_counts().to_dict(),
             "excluded": dict(excluded), "sensor": "Dexcom GL", "step_minutes": 5,
             "history_minutes": [-60, -5], "forecast_minutes": [5, 120]}
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit
