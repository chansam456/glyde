# Glyde

Glyde forecasts how glucose may change over the two hours after a meal. How much can a meal photo and recent glucose readings tell you, and how does that compare with entering the nutrients directly?

On **207 meals from seven people held out of training**, the app’s nutrient model predicted the peak with a **31.4 mg/dL mean absolute error (MAE)** and landed **within ±20 mg/dL on 46.4% of meals**. The photo model gave users a second way to generate a forecast and ran its training and evaluation in **2.51 seconds on CPU** once the image embeddings were cached.

## How it works

- **Build a meal.** Enter the carbohydrate, protein, fat, and fiber for each planned portion, along with a starting glucose reading. The app sums the nutrients and feeds them to an Extra Trees model selected on validation.
- **Use a photo.** Upload a before-meal photo and enter 12 premeal glucose readings at five-minute intervals. A frozen EfficientNet-B0 encodes the image, an LSTM encodes the glucose history, and a shared head combines the two.
- **Explore the forecast.** Both paths use known diabetes status as an input and predict glucose every five minutes from +5 to +120 minutes. The Streamlit app plots the curve, shows its peak and timing, and exports the forecast. The evidence tab lets users inspect model scores and compare predictions against observed curves for held-out meals.

The binary status flag groups healthy and prediabetes participants as 0 and type 2 diabetes participants as 1. The dataset contains no type 1 diabetes cohort.

## How the models were tested

The benchmark uses 1,342 meals from 45 participants in [CGMacros v1.0.0](https://physionet.org/content/cgmacros/1.0.0/). Preparation aligns meal-start photos and nutrient records with Dexcom readings, and excludes overlapping meals, incomplete windows, and invalid inputs. The models use the same eligible meals for comparison.

The seed-42 split is stratified by baseline HbA1c group. Every meal from a participant stays in one partition, so the test measures performance on people the models did not train on.

| Split | Participants | Meals | Healthy / prediabetes / type 2 |
| --- | ---: | ---: | ---: |
| Train | 31 | 923 | 10 / 11 / 10 |
| Validation | 7 | 212 | 3 / 2 / 2 |
| Test | 7 | 207 | 2 / 3 / 2 |

Training data supplies the fitted models and image normalization statistics. Validation peak MAE selects the nutrient estimator and the neural checkpoint. Peak MAE compares the predicted and observed maxima on the two-hour grid. Curve MAE measures error across all 24 forecast points.

| Model | Curve MAE (mg/dL) | Peak MAE (mg/dL) | Peaks within ±20 mg/dL |
| --- | ---: | ---: | ---: |
| Repeat starting glucose | 32.75 | 55.60 | 20.8% |
| Starting glucose + mean training response | 28.61 | 35.09 | 43.0% |
| Nutrients + starting glucose | 25.79 | **31.21** | 44.0% |
| Nutrients + starting glucose + status | **25.75** | 31.40 | **46.4%** |
| Premeal history only (Ridge) | 28.65 | 36.42 | 42.5% |
| Photo + LSTM history, quick | 28.41 | 34.30 | 38.6% |
| Photo + LSTM history + status, quick | 28.51 | 33.73 | 41.1% |

### What the ablations showed

- **Nutrients added signal.** Within Ridge, adding nutrients to starting glucose lowered peak MAE from 36.32 to 33.72 mg/dL. The status-aware Extra Trees model reduced peak MAE by about 11% versus the mean-response baseline. When nutrient amounts are available, this is the stronger path in the app.
- **The diabetes flag changed which errors improved.** Adding the flag raised the share of nutrient forecasts within ±20 mg/dL from 44.0% to 46.4%, while peak MAE rose from 31.21 to 31.40 mg/dL. Training separate nutrient models for the two status groups yielded 31.38 mg/dL and 44.9%. These comparisons reuse the original held-out cohort and are exploratory.
- **The photo model used the image signal.** Zeroing the standardized embedding at inference raised the status-aware photo model’s peak MAE from 33.73 to 46.29 mg/dL. This measures sensitivity to a missing image signal. Estimating the benefit of the photo itself would require retraining a matched history-only LSTM. In the pooled runs, a larger two-layer LSTM gave nearly the same peak error as the quick model (34.27 vs. 34.30 mg/dL).
- **Personalization helped in a separate few-shot experiment.** For the original pooled photo model, calibrating on each person’s first three eligible meals lowered peak MAE from 33.57 to 31.34 mg/dL on the same 186 later meals, a 6.7% reduction. The paired participant bootstrap interval was 1.9% to 10.7%. The status-aware run’s interval included zero.

Caching the frozen embeddings made it practical to rerun neural comparisons on CPU. The 2.51-second benchmark includes fitting, validation, and evaluation with the cache in place. Data and weight downloads, embedding extraction, interpreter startup, and final artifact writes are outside that timing.

The benchmark covers one small cohort and uses a released CGM stream that was already interpolated. App users also estimate portions where the dataset supplies retrospective nutrient annotations. The results describe a research prototype under these conditions.

## Repository layout

| Path | What lives there |
| --- | --- |
| [src/glyde/](src/glyde/) | Data preparation, embedding extraction, nutrient baselines, LSTM training, evaluation, offline calibration, and saved-model inference |
| [app.py](app.py) | Streamlit meal builder, photo upload, forecast plots, and held-out evidence explorer |
| [scripts/](scripts/) | Checksum-verified dataset download and benchmark report generation |
| [reports/](reports/) | Saved metrics, participant split manifests, cohort breakdowns, and neural learning curves |
| [tests/](tests/) | Checks for participant isolation, premeal resampling boundaries, calibration timing, peak metrics, neural gradients, status handling, saved-model inference, and food-entry app behavior |

## Run locally

Use Python 3.11 or newer and run these commands from the repository root.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/download.py
.venv/bin/glyde prepare
.venv/bin/glyde train-fast
.venv/bin/glyde train-status
.venv/bin/streamlit run app.py
```

For the photo tab, run the following from the repository root. The embedding step downloads the pretrained EfficientNet weights on first use and saves the vectors in `data/prepared/`.

```bash
.venv/bin/glyde embed
.venv/bin/glyde train --diabetes-flag
```

Run `.venv/bin/glyde train` for the pooled photo benchmark and `.venv/bin/glyde train --profile scaled --epochs 150 --output artifacts/scaled` for the larger comparison. Checkpoints and per-meal forecasts are written locally to `artifacts/`. Run `.venv/bin/python -m pytest -q` after the nutrient training commands to include the artifact-backed checks.

Project code is MIT-licensed. The dataset is distributed separately under CC BY-NC-SA 4.0.
