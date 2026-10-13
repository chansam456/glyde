from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from glucolens.status import diabetes_flags, validate_flag
from glucolens.inference import predict_food
from glucolens.model import MealLSTM


def test_flag_mapping_and_validation():
    np.testing.assert_equal(diabetes_flags(["healthy", "prediabetes", "type2_diabetes"]), [0, 0, 1])
    for value in [-1, 2, .5, None, "diabetic"]:
        with pytest.raises(ValueError):
            validate_flag(value)
    with pytest.raises(ValueError):
        diabetes_flags(["unknown"])


def test_neural_context_is_required_and_affects_output():
    torch.manual_seed(42)
    model = MealLSTM(context_dim=1).eval()
    history, image = torch.zeros(2, 12, 1), torch.zeros(2, 1280)
    with pytest.raises(ValueError):
        model(history, image)
    result = model(history, image, torch.tensor([[0.], [1.]]))
    assert result.shape == (2, 24)
    assert not torch.allclose(result[0], result[1])


@pytest.mark.skipif(not Path("artifacts/status/food_model.joblib").exists(), reason="Train status models first")
def test_flagged_food_inference_and_locked_splits():
    root = Path("artifacts/status")
    report = json.loads((root / "metrics.json").read_text())
    pooled = json.loads(Path("artifacts/fast/split.json").read_text())
    assert report["split"] == pooled
    rows = pd.read_csv(root / "predictions.csv")
    curves = np.load(root / "curves.npz")
    for flag in [0, 1]:
        i = rows.index[rows.diabetes_flag == flag][0]
        row = rows.iloc[i]
        result = predict_food([row.Carbs, row.Protein, row.Fat, row.Fiber],
            float(curves["history"][i, -1]), artifact=root, diabetes_flag=flag)
        np.testing.assert_allclose(result, curves["binary_flag"][i], rtol=1e-5, atol=1e-4)
    with pytest.raises(ValueError):
        predict_food([60, 25, 15, 8], 100, artifact=root)
