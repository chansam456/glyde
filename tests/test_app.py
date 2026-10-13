"""Smoke checks for real app behavior without a running browser server."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(not Path("artifacts/status/food_model.joblib").exists(), reason="Train status model first")
def test_app_loads_real_artifacts_and_rejects_empty_meal():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Build a meal", "Meal photo", "Model evidence"]
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any("Add at least one nutrient" in warning.value for warning in app.warning)


@pytest.mark.skipif(not Path("artifacts/status/food_model.joblib").exists(), reason="Train status model first")
def test_app_food_entry_generates_forecast():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    # Streamlit's table edits arrive through the widget's documented session payload.
    app.session_state["meal"] = {"edited_rows": {0: {"Food": "Meal", "Carbs (g)": 60.,
        "Protein (g)": 25., "Fat (g)": 15., "Fiber (g)": 8.}}, "added_rows": [], "deleted_rows": []}
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.metric[0].label == "Predicted peak"
    assert app.metric[0].value.endswith("mg/dL")
    peak_non_diabetic = app.metric[0].value
    app.radio[0].set_value(1).run(timeout=30)
    # AppTest cannot synthesize a persistent frontend data-editor interaction;
    # supply the same meal payload for the second status choice.
    app.session_state["meal"] = {"edited_rows": {0: {"Food": "Meal", "Carbs (g)": 60.,
        "Protein (g)": 25., "Fat (g)": 15., "Fiber (g)": 8.}}, "added_rows": [], "deleted_rows": []}
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any("Forecast uses diabetes flag 1" in caption.value for caption in app.caption)
    # These representative inputs should reflect the trained status effect.
    assert app.metric[0].value != peak_non_diabetic
