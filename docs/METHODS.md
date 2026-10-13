# Methods and evaluation protocol

The original protocol below is retained for its pooled benchmark. The current app also uses diabetes-status conditioning, described in [DIABETES_FLAG.md](DIABETES_FLAG.md). Both versions preserve the same participant split and evaluation windows; the new status is an additional known input, not a diagnostic prediction.

## Task and input contract

Forecast the released Dexcom glucose stream at +5, +10, …, +120 minutes relative to a meal timestamp. History consists of 12 values at -60, -55, …, -5 minutes. Peak glucose is the maximum of the forecast grid, not the final reading. The starting glucose is the -5-minute value. V1 takes four nutrient amounts and starting glucose; V2 takes an actual before-meal photograph and the complete premeal history. Neither model receives postmeal activity or future glucose as an input.

V1 uses Ridge (standardized features, alpha 10) and Extra Trees (300 trees, minimum leaf size 8). The choice uses validation peak MAE. V2 uses pretrained frozen EfficientNet-B0 embeddings, a history LSTM, and a 24-output fusion head. AdamW trains the small head with trajectory Smooth L1 loss plus 0.25 times peak Smooth L1 loss, gradient clipping, and early stopping after 15 epochs without validation peak-MAE improvement. Quick profile: hidden size 48, one layer. Scaled profile: hidden size 128, two layers. Pretrained weights are external ImageNet weights, not weights fitted on test meals.

## Preprocessing

Use Dexcom consistently; do not switch sensors to patch missing targets. Normalize column whitespace and use the published `Image path` field. Keep the image recorded at the meal-start row. Do not use after-meal photos to infer consumed quantity.

Interpolate only where the released stream supports the queried time and brackets are no more than 15 minutes apart; reject extrapolation and larger gaps. Historical resampling is restricted to timestamps strictly before meal onset. A next logged meal at or before +120 minutes excludes the current meal. Missing photographs, nutrients, or CGM windows are excluded and counted. Broad, prespecified nutrient bounds exclude negative values or values exceeding 300 g carbohydrate, 300 g protein, 200 g fat, or 100 g fiber. These bounds catch obvious table anomalies, including fiber entries in the hundreds or thousands of grams. Do not silently divide or repair them.

The released glucose table already contains minute-resolution interpolation. Restricting our own resampling to premeal rows does **not** prove that every released historical value was constructed causally from raw sensor observations. This benchmark forecasts the released stream. A prospective deployment would require raw timestamped CGM and causal preprocessing.

The common benchmark requires valid photos and nutrient inputs, even for V1, to make comparisons on the same meals. Expanding a nutrients-only dataset would be a separate experiment with a separate manifest. Macro amounts are retrospective dataset annotations and may reflect consumed portions; the app requires users to estimate planned portions. This creates deployment distribution shift.

## Split contract

Load baseline HbA1c from `bio.csv`, derive the three dataset cohorts, and stratify unique participant IDs. Reserve 15% of participants for testing, then reserve an equally sized validation group from the remainder. Seed 42 yields 31 training, 7 validation, and 7 test participants. Sort participants before splitting so file traversal order cannot change assignments. All participant meals stay together.

The splitting algorithm uses participant IDs and baseline cohort labels, not meal outcomes. Train-only image normalization and model fitting prevent aggregate test statistics from entering predictions. Each artifact directory saves the split, row-level manifest, and participant cohort cross-tab. Small participant numbers prevent exact equality of proportions; report actual counts and cohort-level performance rather than claiming perfectly balanced groups.

## Baselines and image probe

Persistence repeats the last premeal glucose. Population response adds the training participants' mean response curve to the starting value. Glucose-only Ridge in V1 removes the nutrient inputs. History Ridge in V2 uses the 12 premeal readings. Nutrient Ridge and Extra Trees provide stronger structured-data alternatives to the neural model.

The image occlusion probe zeros standardized image embeddings at inference. This is a sensitivity diagnostic, **not a separately trained history-only neural ablation**. Image contribution cannot be established conclusively from this probe. A larger study should train matched history-only, image-only, and multimodal networks under identical budgets.

## Offline calibration experiment

Use only the first three chronologically eligible meals of each test participant to populate that person's residual memory. Store full target-minus-population curves, unit-normalized image embeddings, and target availability at meal onset +120 minutes. Freeze memory and forecast subsequent meals. Apply a softmax-weighted nearest-neighbor residual correction with k=5, temperature 0.1, and base strength 0.5; shrink the strength by the available sample count divided by k. With three calibration meals, maximum gain is 0.3. These fixed settings were not optimized against test improvements.

Compare uncalibrated and calibrated predictions on the same postcalibration meals only. Calibration targets belong to held-out people and support a separate few-shot protocol; this is not the cold-start test setting. Bootstrap 1,000 paired samples of participants, retaining all within-person meals, to produce a 95% interval for relative peak-MAE reduction. Negative reduction means personalization made predictions worse.

## Metrics

- Trajectory MAE and RMSE: error over all 24 future values, in mg/dL.
- Peak MAE: absolute error between predicted and observed grid maxima, in mg/dL.
- Peak within 20 mg/dL: percentage of meals with absolute peak error ≤20 mg/dL. This is tolerance accuracy, not classification precision.
- Peak-time MAE: absolute difference between grid-max times, in minutes; flat curves use the earliest grid maximum.
- Per-person and per-cohort tables: expose unequal meal counts and subgroup performance.

Relative improvement = 100 × (baseline error − model error) / baseline error. Name the baseline when quoting an improvement. A metric alone does not establish clinical usefulness or insulin-sensitivity measurement.

## Scale-up plan

1. Keep the fast model as a mandatory baseline and cache frozen image embeddings once.
2. Use the scaled LSTM profile for more data; compare on validation before using a locked test set.
3. Add repeated stratified participant splits or group cross-validation to quantify sensitivity to the current seven-person test cohort.
4. Add age, baseline HbA1c, BMI, and premeal activity only when the inference workflow can provide them consistently. The current model does not use these features; HbA1c is used only to construct splits.
5. Train matched image/history ablations, perform participant-level learning curves, and investigate photo domain shift.
6. Fine-tune the last EfficientNet block on training participants with validation early stopping only after acquiring a larger dataset. This is future work, not a implemented or measured capability.
7. Add calibrated prediction intervals using a separate participant calibration partition; do not present global MAE as an individual confidence interval.

## Limits

Small cohort, ten days per person, meals concentrated in shakes and restaurant lunches, one benchmark seed, retrospective nutrient labels, potential unlogged meals, sensor interpolation and physiological lag, uncertain medication effects, no prospective external validation, and no measured insulin sensitivity. The two-hour window may miss delayed peaks. Results describe this dataset and protocol only.
