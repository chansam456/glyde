from pathlib import Path
import json
import random
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from .model import MealLSTM
from .personalization import SimilarityMemory
from .metrics import metrics, participant_bootstrap


def split_people(ids, seed=42, cohorts=None):
    if cohorts is None:
        raise ValueError("Participant cohort labels are required for stratified splitting")
    frame = pd.DataFrame({"participant": ids, "cohort": cohorts})
    if frame.groupby("participant").cohort.nunique().max() > 1:
        raise ValueError("A participant cannot have multiple cohort labels")
    frame = frame.drop_duplicates("participant").sort_values("participant")
    trainval, test = train_test_split(frame, test_size=.15, random_state=seed, stratify=frame.cohort)
    train, val = train_test_split(trainval, test_size=len(test), random_state=seed, stratify=trainval.cohort)
    return {name: sorted(group.participant.tolist()) for name, group in
            zip(["train", "validation", "test"], [train, val, test])}


def save_split(meals, split, output):
    output = Path(output)
    (output / "split.json").write_text(json.dumps(split, indent=2))
    manifest = meals[["participant", "cohort", "timestamp", "image"]].copy()
    assignment = {person: key for key, people in split.items() for person in people}
    manifest["split"] = manifest.participant.map(assignment)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    counts = pd.crosstab(manifest.drop_duplicates("participant").split,
                         manifest.drop_duplicates("participant").cohort)
    counts.to_csv(output / "split_cohort_counts.csv")


def food_features(macros, history):
    # Simple inference contract: grams of four nutrients and premeal glucose.
    return np.column_stack([macros, history[:, -1]])


def run(prepared, output, epochs=80, seed=42, profile="quick", max_minutes=45, diabetes_flag=False):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(prepared / "windows.npz")
    meals = pd.read_csv(prepared / "meals.csv")
    h, y, macro = [arrays[x] for x in ["history", "target", "macros"]]
    e = np.load(prepared / "embeddings.npy")
    signature = hashlib.sha256((prepared / "meals.csv").read_bytes()).hexdigest()
    embed_metadata = json.loads((prepared / "embeddings.json").read_text())
    if embed_metadata["meal_manifest_sha256"] != signature or len(e) != len(meals):
        raise ValueError("Embeddings do not match meals; rerun glyde embed")
    split = split_people(meals.participant.to_numpy(), seed, meals.cohort.to_numpy())
    save_split(meals, split, output)
    indices = {key: np.flatnonzero(meals.participant.isin(value)) for key, value in split.items()}
    tr, va, te = [indices[x] for x in ["train", "validation", "test"]]
    baseline = h[:, -1:]
    delta = y - baseline
    xfood = food_features(macro, h)
    food_candidates = {
        "food_ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "food_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, random_state=seed, n_jobs=4),
    }
    predictions = {"persistence": np.repeat(baseline, 24, axis=1)}
    predictions["population_response"] = delta[tr].mean(0)[None, :] + baseline
    history_ridge = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(h[tr], delta[tr])
    predictions["history_ridge"] = history_ridge.predict(h) + baseline
    validation = {}
    for name, model in food_candidates.items():
        model.fit(xfood[tr], delta[tr])
        predictions[name] = model.predict(xfood) + baseline
        validation[name] = metrics(y[va], predictions[name][va])
    food_name = min(food_candidates, key=lambda name: validation[name]["peak_mae_mg_dl"])
    joblib.dump(food_candidates[food_name], output / "food_model.joblib")
    # Normalize all learned inputs using training people only.
    emean, estd = e[tr].mean(0), e[tr].std(0).clip(.01)
    eh = torch.tensor((e - emean) / estd, dtype=torch.float32)
    hh = torch.tensor((h - 120) / 50, dtype=torch.float32).unsqueeze(-1)
    yy = torch.tensor(delta / 50, dtype=torch.float32)
    from .status import diabetes_flags
    context = torch.tensor(diabetes_flags(meals.cohort)).reshape(-1, 1) if diabetes_flag else None
    architecture = {"image_dim": e.shape[1], "hidden": 48 if profile == "quick" else 128,
                    "layers": 1 if profile == "quick" else 2, "context_dim": int(diabetes_flag)}
    model = MealLSTM(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, best_state, best_epoch = float("inf"), None, 0
    learning_curve = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in np.array_split(np.random.permutation(tr), max(1, int(np.ceil(len(tr) / 64)))):
            optimizer.zero_grad()
            prediction = model(hh[batch], eh[batch], context[batch] if context is not None else None)
            loss = torch.nn.functional.smooth_l1_loss(prediction, yy[batch])
            # Explicit peak term complements trajectory reconstruction.
            loss = loss + .25 * torch.nn.functional.smooth_l1_loss(prediction.max(1).values, yy[batch].max(1).values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_pred = model(hh[va], eh[va], context[va] if context is not None else None).numpy() * 50 + baseline[va]
        score = metrics(y[va], val_pred)["peak_mae_mg_dl"]
        learning_curve.append({"epoch": epoch + 1, "loss": np.mean(losses), "validation_peak_mae": score})
        if score < best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: validation peak MAE {score:.2f} mg/dL", flush=True)
        if epoch + 1 - best_epoch >= 15 or time.perf_counter() - started > max_minutes * 60:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions["image_lstm"] = model(hh, eh, context).numpy() * 50 + baseline
        predictions["image_occlusion_probe"] = model(hh, torch.zeros_like(eh), context).numpy() * 50 + baseline
    validation["image_lstm"] = metrics(y[va], predictions["image_lstm"][va])
    # Offline calibration: fit memory on the first three meals, then freeze it.
    personalized = predictions["image_lstm"].copy()
    memories_used = np.zeros(len(meals), dtype=int)
    post_calibration = []
    for participant in split["test"]:
        memory = SimilarityMemory()
        ordered = meals[meals.participant == participant].sort_values("timestamp").index.to_numpy()
        for i in ordered[:3]:
            memory.observe(e[i], predictions["image_lstm"][i], y[i], pd.Timestamp(meals.available_at[i]))
        for i in ordered[3:]:
            now = pd.Timestamp(meals.timestamp[i])
            personalized[i], memories_used[i] = memory.correct(e[i], predictions["image_lstm"][i], now)
            post_calibration.append(i)
    predictions["offline_personalized_image_lstm"] = personalized
    pc = np.array(post_calibration, dtype=int)
    results = {name: metrics(y[te], pred[te]) for name, pred in predictions.items() if name != "offline_personalized_image_lstm"}
    calibrated_results = {name: metrics(y[pc], pred[pc]) for name, pred in predictions.items()}
    # Macro metrics treat each participant equally to expose cohort imbalance.
    by_person = []
    for participant in split["test"]:
        idx = np.flatnonzero(meals.participant == participant)
        for name, pred in predictions.items():
            if name != "offline_personalized_image_lstm":
                by_person.append({"participant": participant, "model": name, **metrics(y[idx], pred[idx])})
    person_frame = pd.DataFrame(by_person)
    error_frame = pd.DataFrame({"participant": meals.participant.iloc[pc].to_numpy(),
        "base_peak_error": np.abs(y[pc].max(1) - predictions["image_lstm"][pc].max(1)),
        "personal_peak_error": np.abs(y[pc].max(1) - personalized[pc].max(1))})
    base = calibrated_results["image_lstm"]["peak_mae_mg_dl"]
    improvement = 100 * (base - calibrated_results["offline_personalized_image_lstm"]["peak_mae_mg_dl"]) / max(base, 1e-8)
    report = {"seed": seed, "split": split, "split_meals": {k: len(v) for k, v in indices.items()},
        "profile": profile, "architecture": architecture,
        "uses_diabetes_flag": diabetes_flag,
        "epochs_run": len(learning_curve), "best_epoch": best_epoch, "selected_food_model": food_name,
        "validation": validation, "test": results,
        "offline_calibration": {"calibration_meals_per_person": 3, "memory_frozen": True,
                                "evaluation": calibrated_results},
        "participant_macro_peak_mae": person_frame.groupby("model").peak_mae_mg_dl.mean().to_dict(),
        "personalization_peak_mae_reduction_pct": improvement,
        "personalization_reduction_95pct_ci": participant_bootstrap(error_frame),
        "training_evaluation_seconds": time.perf_counter() - started,
        "status": "single-seed research benchmark; no clinical validation"}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    (output / "split.json").write_text(json.dumps(split, indent=2))
    pd.DataFrame(learning_curve).to_csv(output / "learning_curve.csv", index=False)
    person_frame.to_csv(output / "participant_metrics.csv", index=False)
    cohort_rows = []
    for cohort in sorted(meals.cohort.unique()):
        idx = te[meals.iloc[te].cohort.to_numpy() == cohort]
        for name, pred in predictions.items():
            if name != "offline_personalized_image_lstm":
                cohort_rows.append({"cohort": cohort, "model": name, **metrics(y[idx], pred[idx])})
    pd.DataFrame(cohort_rows).to_csv(output / "cohort_metrics.csv", index=False)
    torch.save({"state_dict": best_state, "architecture": architecture, "embedding_mean": torch.tensor(emean),
                "embedding_std": torch.tensor(estd), "seed": seed}, output / "image_lstm.pt")
    test_rows = meals.iloc[te].copy()
    test_rows["memory_meals"] = memories_used[te]
    test_rows["actual_peak"] = y[te].max(1)
    for name, pred in predictions.items():
        test_rows[name + "_peak"] = pred[te].max(1)
    test_rows.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "curves.npz", target=y[te], history=h[te], **{k: v[te] for k, v in predictions.items()})
    return report
