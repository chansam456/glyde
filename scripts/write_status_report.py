"""Document the binary-status extension without overwriting the original benchmark."""
import json
from pathlib import Path
import shutil

food = json.loads(Path("artifacts/status/metrics.json").read_text())
photo = json.loads(Path("artifacts/multimodal-status/metrics.json").read_text())
original = json.loads(Path("artifacts/multimodal/metrics.json").read_text())
lines = ["| Nutrient model | Peak MAE (mg/dL) | Peaks within ±20 mg/dL | Curve MAE (mg/dL) |",
         "|---|---:|---:|---:|"]
for name, score in food["test"].items():
    lines.append(f"| {name} | {score['peak_mae_mg_dl']:.2f} | {score['peak_within_20_mg_dl_pct']:.1f}% | {score['trajectory_mae_mg_dl']:.2f} |")
table = "\n".join(lines)
cohort_lines = ["| Cohort | Held-out meals | Pooled peak MAE | Binary-flag peak MAE | Separate-model peak MAE |",
                "|---|---:|---:|---:|---:|"]
for cohort, models in food["cohort_test"].items():
    cohort_lines.append(f"| {cohort} | {models['pooled']['meals']} | {models['pooled']['peak_mae_mg_dl']:.2f} | {models['binary_flag']['peak_mae_mg_dl']:.2f} | {models['separate_models']['peak_mae_mg_dl']:.2f} |")
cohort_table = "\n".join(cohort_lines)
p = photo["test"]["image_lstm"]
q = original["test"]["image_lstm"]
text = f"""# Binary diabetes flag extension

Implemented October 8, 2026, following the request to distinguish diabetic and non-diabetic responses with a binary input.

## Meaning of the flag

- **0: non-diabetic** — healthy and prediabetes CGMacros participants.
- **1: type 2 diabetic** — CGMacros type 2 diabetes participants.

Labels come from the existing baseline HbA1c-derived dataset cohorts. This is a known-status input, not a classifier or a diagnostic output. The app asks the user to select their known status. Prediabetes is pooled into flag 0 for this binary experiment but remains separately reported. The dataset contains no type 1 diabetes cohort; this model is not evaluated for type 1 diabetes.

## Implementation

The nutrient model receives a sixth feature after carbohydrate, protein, fat, fiber, and starting glucose: diabetes_flag. The photo model concatenates the binary value with the EfficientNet embedding projection and LSTM history representation before the forecast head. Both produce the same 24-point two-hour trajectory as before.

The app's sidebar selector feeds both prediction paths. Saved flag-enabled models require a valid 0/1 input; missing or invalid flags are rejected. Original pooled checkpoints remain usable through their previous inference interface, and reject status arguments rather than silently ignoring them.

For comparison, training also creates separate diabetic and non-diabetic nutrient models. These are research comparators. The default app uses the shared binary-feature model, allowing it to share nutrient-response information across groups. Ridge versus Extra Trees selection uses validation peak MAE separately within each model family and group. All selected nutrient estimators were Extra Trees in this run. The input flag is included by design; no decision to include it was made using test scores.

## Preserve the original split

All models reuse the exact seed-42 participant assignments from the original stratified benchmark: 31 train, 7 validation, 7 test; 923 / 212 / 207 meals. No participant moves between splits and no new row-level split is created. Subgroup models filter these fixed partitions:

| Status | Train people / meals | Validation people / meals | Test people / meals |
|---|---:|---:|---:|
| Non-diabetic | 21 / 634 | 5 / 155 | 5 / 157 |
| Type 2 diabetic | 10 / 289 | 2 / 57 | 2 / 50 |

The test cohort is reused for this user-requested feature extension after earlier results were observed. Results are an exploratory comparison, not a new untouched confirmatory study. Hyperparameters remain fixed and checkpoint/model selection uses validation.

## Measured nutrient-model results

{table}

Adding the flag improves tolerance accuracy from 44.0% to 46.4% (**2.4 percentage points**), but slightly increases peak MAE from 31.21 to 31.40 mg/dL. It does not demonstrate a meaningful overall peak-MAE improvement. Separate models perform similarly; splitting the data by status does not automatically produce a better predictor.

### Report the original cohorts separately

{cohort_table}

Errors are in mg/dL. The diabetic test group contains only two people, so subgroup differences are uncertain. Preserve these subgroup tables when discussing model performance.

## Measured photo-model results

| Photo model | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |
|---|---:|---:|
| Original pooled EfficientNet + LSTM | {q['peak_mae_mg_dl']:.2f} | {q['peak_within_20_mg_dl_pct']:.1f}% |
| EfficientNet + LSTM with binary status | {p['peak_mae_mg_dl']:.2f} | {p['peak_within_20_mg_dl_pct']:.1f}% |

Photo-model initialization changes because its head has one additional input. The difference is not a causal estimate of the flag's benefit; repeated seeds and matched ablations would be needed. Offline personalization is also evaluated for the new photo checkpoint in its metrics file; the original 6.7% result belongs to the original checkpoint and should not be attributed to this extension.

## Training and reproduction

The nutrient comparison trained/evaluated in {food['training_evaluation_seconds']:.2f} seconds; the flag-enabled neural run took {photo['training_evaluation_seconds']:.2f} seconds, excluding interpreter startup and previously cached image extraction. Both remain well below the one-hour training budget.

```bash
rtk proxy .venv/bin/glyde train-status
rtk proxy .venv/bin/glyde train --diabetes-flag
rtk proxy .venv/bin/python scripts/write_status_report.py
rtk proxy .venv/bin/python -m pytest -q
rtk proxy .venv/bin/streamlit run app.py
```

Artifacts: artifacts/status/ contains the shared binary-feature model, pooled comparator, separate diabetic/non_diabetic models, curves and split manifests. artifacts/multimodal-status/ contains the flag-enabled photo checkpoint. Source-controlled aggregate evidence is in reports/status/ and reports/multimodal-status/. The older benchmark remains in [RESULTS.md](RESULTS.md).

## Resume wording for this extension

“Added binary diabetes-status conditioning to nutrient and EfficientNet/LSTM glucose forecasters; compared shared and subgroup models using fixed participant-stratified splits, achieving 46.4% of peak predictions within ±20 mg/dL on 207 held-out meals.”

Do not describe 46.4% as diabetes-classification accuracy or claim that the binary flag established improved clinical accuracy.
"""
Path("docs/DIABETES_FLAG.md").write_text(text)
for directory in ["status", "multimodal-status"]:
    target = Path("reports") / directory
    target.mkdir(parents=True, exist_ok=True)
    source = Path("artifacts") / directory
    for file in ["metrics.json", "split.json", "split_cohort_counts.csv", "cohort_metrics.csv", "learning_curve.csv"]:
        if (source / file).exists():
            shutil.copyfile(source / file, target / file)
print("Wrote docs/DIABETES_FLAG.md and aggregate status reports")
