# Measured results

Benchmark completed October 8, 2026, using CGMacros v1.0.0, seed 42, 1,342 eligible meals, and all 45 participants. Numbers below are real saved-model results. The cohort split supersedes the initial exploratory unstratified run. This is one small held-out cohort, not external validation.

## Stratified split

| Split | Healthy | Prediabetes | Type 2 diabetes | Participants | Meals |
|---|---:|---:|---:|---:|---:|
| train | 10 | 11 | 10 | 31 | 923 |
| validation | 3 | 2 | 2 | 7 | 212 |
| test | 2 | 3 | 2 | 7 | 207 |

All meals from each participant stay together. HbA1c is used to stratify, not as a prediction feature. Exact IDs are saved in [split.json](../reports/fast/split.json); each model writes the complete local meal manifest to its artifact directory. Both versions use identical participants and eligible meals.

Excluded records: {"missing_start_photo": 51, "overlapping_meal": 251, "missing_macros": 1, "incomplete_cgm": 49, "implausible_macros": 12}. Exclusions are counted in sequential order, so these are mutually exclusive primary reasons. Nutrient plausibility checks removed 12 obvious anomalies without guessing corrected values.

## Cold-start test: 207 meals from seven unseen people

### V1 food nutrients and starting glucose

| Model | Curve MAE (mg/dL) | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |
|---|---:|---:|---:|
| ridge | 27.80 | 33.72 | 44.4% |
| extra_trees | 25.79 | 31.21 | 44.0% |
| persistence | 32.75 | 55.60 | 20.8% |
| population_response | 28.61 | 35.09 | 43.0% |
| glucose_only_ridge | 28.78 | 36.32 | 44.0% |

Validation selected **extra_trees**. Peak MAE is **31.21 mg/dL**, a **11.0% reduction versus the population-response baseline**, and **43.9% versus persistence**. Its tolerance accuracy is **44.0% within ±20 mg/dL**. These percentages describe different metrics and should not be called precision.

### V2 quick: EfficientNet embeddings plus LSTM history

| Model | Curve MAE (mg/dL) | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |
|---|---:|---:|---:|
| history_ridge | 28.65 | 36.42 | 42.5% |
| image_lstm | 28.41 | 34.30 | 38.6% |
| image_occlusion_probe | 30.78 | 46.90 | 29.0% |

Best checkpoint: epoch 2 of 17 executed epochs. Early stopping selected the checkpoint on validation. The nutrient model achieves lower held-out peak error than this photo model. The zero-embedding probe is an inference sensitivity diagnostic, not a trained neural ablation.

### V2 scaled profile

| Model | Curve MAE (mg/dL) | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |
|---|---:|---:|---:|
| history_ridge | 28.65 | 36.42 | 42.5% |
| image_lstm | 28.91 | 34.27 | 38.6% |

Scaled configuration: two LSTM layers with 128 hidden units; best checkpoint epoch 2 of 17. This optional profile was evaluated with the same predefined settings. A larger network is not guaranteed to improve this small dataset. The app retains the documented quick photo model; scaled results are reported separately rather than replacing it based on test performance.

## Offline personalization: separate few-shot protocol

Each of the seven held-out people contributes their first three eligible meals for calibration. The remaining 186 meals are the shared evaluation set. Memory is frozen; later outcomes never update it.

| Model | Curve MAE (mg/dL) | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |
|---|---:|---:|---:|
| image_lstm | 27.46 | 33.57 | 39.8% |
| offline_personalized_image_lstm | 25.50 | 31.34 | 40.3% |
| food_trees | 25.06 | 30.83 | 45.2% |

For the quick photo model, offline calibration reduces peak MAE from **33.57 to 31.34 mg/dL**, a **6.7% relative reduction**. The paired participant bootstrap 95% interval is **1.9% to 10.7%**, using 1,000 resamples of seven participants. This interval is exploratory and depends strongly on the small cohort. Personalization adapts residual response curves; it does not measure insulin sensitivity. It does not beat the nutrient model on this shared subset.

The scaled model's calibration change is 5.0% with interval -3.0% to 13.6%; its interval includes zero, so the scaled personalization benefit is uncertain.

## Training time

Environment: Python 3.13.5, PyTorch 2.14.1, Darwin arm64, CPU execution, four neural-training threads.

| Model | Training and evaluation | Epochs run | Best epoch |
|---|---:|---:|---:|
| V1 nutrient model | 0.29 seconds | N/A | N/A |
| V2 quick | 2.61 seconds | 17 | 2 |
| V2 scaled | 5.09 seconds | 17 | 2 |

Timing starts inside each training routine and includes loading prepared inputs, fitting, validation, test evaluation, and bootstrap where applicable. It excludes interpreter startup, downloading data/weights, image embedding extraction, and final artifact writes. These model-training runs are comfortably below one hour on this machine. EfficientNet is frozen, so only its small downstream model is trained. Embedding extraction completed in the session; an exact extraction runtime was not captured for this cache. Future extraction runs record it in embeddings.json. Network downloads are one-time setup costs.

## Verification

- Core safeguards: seven passing tests for split disjointness and cohort proportions, missing data, causal resampling boundaries, timestamp-gated residual calibration, similar-meal lookup, peak metrics, and gradients through both model branches.
- Saved nutrient-model inference matches the benchmark forecast for an actual held-out meal.
- Saved photo-model inference from the original held-out photo and history matches its cached benchmark forecast to numerical tolerance.
- App checks load actual artifacts, inspect all three tabs, reject empty meals, and exercise a nutrient-based forecast. The final test outcome is recorded in WORK_LOG.md.
- Participant/cohort aggregate tables and source-controlled metrics live in reports/; full forecasts, local image references, and checkpoints live in artifacts/.

## Interpretation

This project demonstrates a complete evaluated pipeline and honest model comparison. The strongest current default is the nutrient model. The photo model and offline calibration provide a working multimodal research extension. Additional participant-stratified runs, prospective raw-CGM data, external cohorts, and calibrated uncertainty are required before stronger generalization claims. See [METHODS.md](METHODS.md) for limitations and scale-up steps.
