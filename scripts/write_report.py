"""Generate portfolio documentation from measured benchmark artifacts."""
from pathlib import Path
import json
import platform
import shutil
import pandas as pd
import torch


def load(name):
    return json.loads((Path("artifacts") / name / "metrics.json").read_text())


def table(metrics):
    lines = ["| Model | Curve MAE (mg/dL) | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |",
             "|---|---:|---:|---:|"]
    for name, result in metrics.items():
        lines.append(f"| {name} | {result['trajectory_mae_mg_dl']:.2f} | {result['peak_mae_mg_dl']:.2f} | {result['peak_within_20_mg_dl_pct']:.1f}% |")
    return "\n".join(lines)


fast, photo, scaled = [load(name) for name in ["fast", "multimodal", "scaled"]]
audit = json.loads(Path("data/prepared/audit.json").read_text())
counts = pd.read_csv("artifacts/fast/split_cohort_counts.csv").set_index("split")
selection = fast["selected_model"]
best = fast["test"][selection]
population = fast["test"]["population_response"]["peak_mae_mg_dl"]
persistence = fast["test"]["persistence"]["peak_mae_mg_dl"]
relative = (population - best["peak_mae_mg_dl"]) / population * 100
relative_persistence = (persistence - best["peak_mae_mg_dl"]) / persistence * 100
personal = photo["offline_calibration"]["evaluation"]
ci = photo["personalization_reduction_95pct_ci"]
docs = Path("docs")
docs.mkdir(exist_ok=True)
reports = Path("reports")
reports.mkdir(exist_ok=True)
for name in ["fast", "multimodal", "scaled"]:
    source = Path("artifacts") / name
    target = reports / name
    target.mkdir(exist_ok=True)
    # Source-controlled evidence contains aggregate metrics and participant splits,
    # not private local paths, original photographs, or per-meal timestamps.
    for file in ["metrics.json", "split.json", "split_cohort_counts.csv", "cohort_metrics.csv", "learning_curve.csv"]:
        if (source / file).exists():
            shutil.copyfile(source / file, target / file)
shutil.copyfile("data/prepared/audit.json", reports / "data_audit.json")
for path in reports.rglob("metrics.json"):
    report = json.loads(path.read_text())
    report.get("participant_macro_peak_mae", {}).pop("offline_personalized_image_lstm", None)
    path.write_text(json.dumps(report, indent=2))
split_lines = ["| Split | Healthy | Prediabetes | Type 2 diabetes | Participants | Meals |",
               "|---|---:|---:|---:|---:|---:|"]
for split in ["train", "validation", "test"]:
    row = counts.loc[split]
    split_lines.append(f"| {split} | {row.healthy} | {row.prediabetes} | {row.type2_diabetes} | {int(row.sum())} | {fast['split_meals'][split]} |")
split_table = "\n".join(split_lines)
runtime_lines = []
for label, report in [("V1 nutrient model", fast), ("V2 quick", photo), ("V2 scaled", scaled)]:
    runtime_lines.append(f"| {label} | {report['training_evaluation_seconds']:.2f} seconds | {report.get('epochs_run', 'N/A')} | {report.get('best_epoch', 'N/A')} |")
runtime_table = "\n".join(runtime_lines)
result = f"""# Measured results

Benchmark completed October 8, 2026, using CGMacros v1.0.0, seed 42, {audit['eligible_meals']:,} eligible meals, and all 45 participants. Numbers below are real saved-model results. The cohort split supersedes the initial exploratory unstratified run. This is one small held-out cohort, not external validation.

## Stratified split

{split_table}

All meals from each participant stay together. HbA1c is used to stratify, not as a prediction feature. Exact IDs are saved in [split.json](../reports/fast/split.json); each model writes the complete local meal manifest to its artifact directory. Both versions use identical participants and eligible meals.

Excluded records: {json.dumps(audit['excluded'])}. Exclusions are counted in sequential order, so these are mutually exclusive primary reasons. Nutrient plausibility checks removed 12 obvious anomalies without guessing corrected values.

## Cold-start test: 207 meals from seven unseen people

### V1 food nutrients and starting glucose

{table(fast['test'])}

Validation selected **{selection}**. Peak MAE is **{best['peak_mae_mg_dl']:.2f} mg/dL**, a **{relative:.1f}% reduction versus the population-response baseline**, and **{relative_persistence:.1f}% versus persistence**. Its tolerance accuracy is **{best['peak_within_20_mg_dl_pct']:.1f}% within ±20 mg/dL**. These percentages describe different metrics and should not be called precision.

### V2 quick: EfficientNet embeddings plus LSTM history

{table({k: photo['test'][k] for k in ['history_ridge', 'image_lstm', 'image_occlusion_probe']})}

Best checkpoint: epoch {photo['best_epoch']} of {photo['epochs_run']} executed epochs. Early stopping selected the checkpoint on validation. The nutrient model achieves lower held-out peak error than this photo model. The zero-embedding probe is an inference sensitivity diagnostic, not a trained neural ablation.

### V2 scaled profile

{table({k: scaled['test'][k] for k in ['history_ridge', 'image_lstm']})}

Scaled configuration: two LSTM layers with 128 hidden units; best checkpoint epoch {scaled['best_epoch']} of {scaled['epochs_run']}. This optional profile was evaluated with the same predefined settings. A larger network is not guaranteed to improve this small dataset. The app retains the documented quick photo model; scaled results are reported separately rather than replacing it based on test performance.

## Offline personalization: separate few-shot protocol

Each of the seven held-out people contributes their first three eligible meals for calibration. The remaining {personal['image_lstm']['meals']} meals are the shared evaluation set. Memory is frozen; later outcomes never update it.

{table({k: personal[k] for k in ['image_lstm', 'offline_personalized_image_lstm', 'food_trees']})}

For the quick photo model, offline calibration reduces peak MAE from **{personal['image_lstm']['peak_mae_mg_dl']:.2f} to {personal['offline_personalized_image_lstm']['peak_mae_mg_dl']:.2f} mg/dL**, a **{photo['personalization_peak_mae_reduction_pct']:.1f}% relative reduction**. The paired participant bootstrap 95% interval is **{ci[0]:.1f}% to {ci[1]:.1f}%**, using 1,000 resamples of seven participants. This interval is exploratory and depends strongly on the small cohort. Personalization adapts residual response curves; it does not measure insulin sensitivity. It does not beat the nutrient model on this shared subset.

The scaled model's calibration change is {scaled['personalization_peak_mae_reduction_pct']:.1f}% with interval {scaled['personalization_reduction_95pct_ci'][0]:.1f}% to {scaled['personalization_reduction_95pct_ci'][1]:.1f}%; its interval includes zero, so the scaled personalization benefit is uncertain.

## Training time

Environment: Python {platform.python_version()}, PyTorch {torch.__version__}, {platform.system()} {platform.machine()}, CPU execution, four neural-training threads.

| Model | Training and evaluation | Epochs run | Best epoch |
|---|---:|---:|---:|
{runtime_table}

Timing starts inside each training routine and includes loading prepared inputs, fitting, validation, test evaluation, and bootstrap where applicable. It excludes interpreter startup, downloading data/weights, image embedding extraction, and final artifact writes. These model-training runs are comfortably below one hour on this machine. EfficientNet is frozen, so only its small downstream model is trained. Embedding extraction completed in the session; an exact extraction runtime was not captured for this cache. Future extraction runs record it in embeddings.json. Network downloads are one-time setup costs.

## Verification

- Core safeguards: seven passing tests for split disjointness and cohort proportions, missing data, causal resampling boundaries, timestamp-gated residual calibration, similar-meal lookup, peak metrics, and gradients through both model branches.
- Saved nutrient-model inference matches the benchmark forecast for an actual held-out meal.
- Saved photo-model inference from the original held-out photo and history matches its cached benchmark forecast to numerical tolerance.
- App checks load actual artifacts, inspect all three tabs, reject empty meals, and exercise a nutrient-based forecast. The final test outcome is recorded in WORK_LOG.md.
- Participant/cohort aggregate tables and source-controlled metrics live in reports/; full forecasts, local image references, and checkpoints live in artifacts/.

## Interpretation

This project demonstrates a complete evaluated pipeline and honest model comparison. The strongest current default is the nutrient model. The photo model and offline calibration provide a working multimodal research extension. Additional participant-stratified runs, prospective raw-CGM data, external cohorts, and calibrated uncertainty are required before stronger generalization claims. See [METHODS.md](METHODS.md) for limitations and scale-up steps.
"""
(docs / "RESULTS.md").write_text(result)
(docs / "RESUME.md").write_text(f"""# Resume-ready description

## Project title

**Glucolens — Multimodal Meal-to-Glucose Forecasting**  
Python · PyTorch · EfficientNet · LSTM · scikit-learn · Streamlit

## Recommended bullets

- Built an end-to-end meal-to-glucose forecasting pipeline using {audit['eligible_meals']:,} meals from 45 participants, achieving **{best['peak_mae_mg_dl']:.1f} mg/dL peak MAE** and **{relative:.1f}% lower peak error than a population-response baseline** on 207 meals from seven held-out participants.
- Implemented **EfficientNet-B0 + LSTM** forecasting and offline similarity-based personalization, reducing peak MAE by **{photo['personalization_peak_mae_reduction_pct']:.1f}%** after three calibration meals per person, with participant-stratified healthy/prediabetes/type 2 diabetes splits.
- Delivered a local meal-entry and photo-prediction app with saved-model inference, reproducible checkpoints, cohort-level evaluation, and cached embeddings; trained the nutrient model in **{fast['training_evaluation_seconds']:.2f} seconds** and the quick neural model in **{photo['training_evaluation_seconds']:.2f} seconds**, excluding one-time embedding extraction.

Use two bullets if space is tight. The first metric belongs to the nutrient model, not the image model. The 6.7% personalization result belongs to the quick photo model and its separate 186-meal postcalibration evaluation. These are local single-seed benchmark results, not production deployment results.

## Interview explanation

Meal nutrients are the strongest current predictor. I tested the more complex image/history model against structured-data and glucose baselines rather than assuming it would win. For unseen participants, a small frozen calibration set corrects systematic individual residuals using visually similar meals. I prevent person overlap across partitions, choose checkpoints using validation, and show cohort results and participant-bootstrap uncertainty. The main engineering contribution is the complete reproducible system and fair comparison.

## Claims to avoid

Do not claim classification precision, measured insulin sensitivity, prospective clinical accuracy, an image model outperforming the nutrient model, automatic nutrient extraction from arbitrary food names, production deployment, or a statistically established scalable-model personalization improvement. All reported percentages must name their metric and comparison baseline.
""")
print("Wrote docs/RESULTS.md, docs/RESUME.md and aggregate reports/")
