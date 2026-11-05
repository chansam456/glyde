"""Run: .venv/bin/streamlit run app.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from glyde.inference import predict_food, predict_image

st.set_page_config(page_title="Glyde", page_icon="🥗", layout="wide")
st.title("Glyde")
st.caption("Explore how a meal may affect the next two hours of glucose.")
st.info("Research prototype trained on CGMacros. Predictions are estimates, not guidance for medication or insulin dosing.")
diabetes_flag = st.sidebar.radio("Diabetes status", [0, 1],
    format_func=lambda value: "Non-diabetic (0)" if value == 0 else "Type 2 diabetic (1)")
st.sidebar.caption("Choose your known status. In this dataset, flag 0 includes healthy and prediabetes participants; flag 1 is type 2 diabetes. Type 1 diabetes was not studied.")
food_artifact = Path("artifacts/status")
photo_artifact = Path("artifacts/multimodal-status")
food_tab, photo_tab, evidence_tab = st.tabs(["Build a meal", "Meal photo", "Model evidence"])


def display_curve(curve, baseline):
    a, b, c = st.columns(3)
    a.metric("Predicted peak", f"{curve.max():.0f} mg/dL")
    b.metric("Peak change from start", f"{curve.max() - baseline:+.0f} mg/dL")
    c.metric("Time of predicted peak", f"{(curve.argmax() + 1) * 5} min")
    frame = pd.DataFrame({"Minutes after meal": np.arange(5, 121, 5), "Glucose (mg/dL)": curve})
    st.line_chart(frame.set_index("Minutes after meal"))
    st.download_button("Download forecast", frame.to_csv(index=False), "glyde_forecast.csv", "text/csv")


with food_tab:
    st.subheader("Add foods and their nutrients")
    st.write("Enter nutrient amounts for the portion you plan to eat, from food labels or a nutrition database. Food names are labels; predictions use the nutrient totals.")
    edited = st.data_editor(pd.DataFrame([{"Food": "", "Carbs (g)": 0., "Protein (g)": 0., "Fat (g)": 0., "Fiber (g)": 0.}]),
        num_rows="dynamic", hide_index=True, key="meal",
        column_config={c: st.column_config.NumberColumn(min_value=0., max_value=500.) for c in ["Carbs (g)", "Protein (g)", "Fat (g)", "Fiber (g)"]})
    glucose = st.number_input("Starting glucose (mg/dL)", 40., 400., 100., step=1.)
    st.caption("Use a recent premeal reading. The benchmark uses the reading five minutes before the meal. Changing this value changes the forecast.")
    totals = edited[["Carbs (g)", "Protein (g)", "Fat (g)", "Fiber (g)"]].fillna(0).sum().to_numpy()
    st.write(f"Meal totals: {totals[0]:.1f} g carbohydrate · {totals[1]:.1f} g protein · {totals[2]:.1f} g fat · {totals[3]:.1f} g fiber")
    ready = (food_artifact / "food_model.joblib").exists()
    if not ready:
        st.warning("Train the binary-status nutrient model first; see the README.")
    if st.button("Predict meal response", disabled=not ready):
        if totals.sum() == 0:
            st.warning("Add at least one nutrient amount before predicting.")
        else:
            report = json.loads((food_artifact / "metrics.json").read_text())
            features = np.r_[totals, glucose, diabetes_flag]
            if ((features < report["training_feature_min"]) | (features > report["training_feature_max"])).any():
                st.warning("Some inputs fall outside the training range; this forecast may be unreliable.")
            display_curve(predict_food(totals, glucose, artifact=food_artifact, diabetes_flag=diabetes_flag), glucose)
            st.caption(f"Forecast uses diabetes flag {diabetes_flag}.")
            st.caption(f"Held-out peak MAE: {report['test'][report['selected_model']]['peak_mae_mg_dl']:.1f} mg/dL. This average error is not an individual confidence interval.")

with photo_tab:
    st.subheader("Forecast from a meal photo and recent glucose")
    photo = st.file_uploader("Upload a photo taken before eating", type=["jpg", "jpeg", "png"])
    history_text = st.text_input("12 readings, comma separated (oldest to newest)", placeholder="Readings at -60, -55, …, -5 minutes")
    st.caption("The photo model requires actual premeal history. Uploads are processed locally and are not saved.")
    if photo:
        st.image(photo, width=360)
    if st.button("Predict from photo", disabled=not (photo and (photo_artifact / "image_lstm.pt").exists())):
        try:
            history = [float(v.strip()) for v in history_text.split(",")]
            curve = predict_image(Image.open(photo), history, artifact=photo_artifact, diabetes_flag=diabetes_flag)
            display_curve(curve, history[-1])
        except (ValueError, OSError) as error:
            st.error(str(error))

with evidence_tab:
    for label, directory in [("Nutrient model: binary flag vs pooled vs separate models", "status"),
                             ("Photo + LSTM with diabetes flag", "multimodal-status"),
                             ("Original pooled nutrient baseline", "fast")]:
        path = Path("artifacts") / directory
        if not (path / "metrics.json").exists():
            continue
        report = json.loads((path / "metrics.json").read_text())
        st.subheader(label)
        st.dataframe(pd.DataFrame(report["test"]).T, width="stretch")
        st.caption(f"Split by participant: {report['split_meals']}. One fixed seed; generalization remains uncertain.")
        if "offline_calibration" in report:
            with st.expander("Offline personalization: three earlier meals per person"):
                evaluation = report["offline_calibration"]["evaluation"]
                names = ["image_lstm", "offline_personalized_image_lstm", "food_trees"]
                st.dataframe(pd.DataFrame({name: evaluation[name] for name in names}).T, width="stretch")
                low, high = report["personalization_reduction_95pct_ci"]
                st.caption(f"Peak error reduction: {report['personalization_peak_mae_reduction_pct']:.1f}% versus the same photo model; participant bootstrap interval {low:.1f}% to {high:.1f}%. Calibration meals are excluded.")
        with st.expander("Participant groups and cohort performance"):
            st.dataframe(pd.read_csv(path / "split_cohort_counts.csv"), hide_index=True)
            st.dataframe(pd.read_csv(path / "cohort_metrics.csv"), hide_index=True)
        if (path / "curves.npz").exists():
            curves = np.load(path / "curves.npz")
            meals = pd.read_csv(path / "predictions.csv")
            sample = st.slider("Inspect held-out meal", 0, len(meals) - 1, key=directory)
            st.write(f"Participant {meals.iloc[sample].participant} · {meals.iloc[sample].meal_type}")
            selected = report.get("selected_model", "image_lstm")
            st.line_chart(pd.DataFrame({"Observed": curves['target'][sample], "Predicted": curves[selected][sample]}, index=np.arange(5, 121, 5)))
