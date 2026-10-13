import numpy as np
import pandas as pd
import torch
from glucolens.data import sample_window
from glucolens.experiment import split_people
from glucolens.metrics import metrics
from glucolens.model import MealLSTM
from glucolens.personalization import SimilarityMemory


def test_window_rejects_long_gap_and_extrapolation():
    times = pd.to_datetime(["2024-01-01 12:00", "2024-01-01 12:30"])
    series = pd.Series([100., 160.], index=times)
    assert sample_window(series, [pd.Timestamp("2024-01-01 12:15")]) is None
    assert sample_window(series, [pd.Timestamp("2024-01-01 11:59")]) is None
    np.testing.assert_equal(sample_window(series, times), [100., 160.])


def test_historical_cutoff_does_not_use_future_spike():
    times = pd.date_range("2024-01-01 11:00", periods=13, freq="5min")
    series = pd.Series([100.] * 12 + [400.], index=times)
    onset = times[-1]
    np.testing.assert_equal(sample_window(series[series.index < onset], times[:-1]), [100.] * 12)


def test_people_never_cross_partitions():
    labels = np.array(["healthy"] * 15 + ["prediabetes"] * 16 + ["type2_diabetes"] * 14)
    split = split_people(np.repeat(np.arange(45), 10), cohorts=np.repeat(labels, 10))
    sets = [set(v) for v in split.values()]
    assert len(set.union(*sets)) == 45
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert split == split_people(np.arange(45), cohorts=labels)
    for people in split.values():
        assert set(labels[people]) == set(labels)
        proportions = pd.Series(labels[people]).value_counts(normalize=True)
        overall = pd.Series(labels).value_counts(normalize=True)
        assert (proportions - overall).abs().max() < .12


def test_calibration_cannot_read_unavailable_targets():
    memory = SimilarityMemory(k=1, strength=1)
    embedding = np.array([1., 0.])
    prediction = np.array([100., 110.])
    memory.observe(embedding, prediction, prediction + 20, available_at=120)
    untouched, count = memory.correct(embedding, prediction, now=119)
    np.testing.assert_equal(untouched, prediction)
    assert count == 0
    corrected, count = memory.correct(embedding, prediction, now=120)
    np.testing.assert_equal(corrected, prediction + 20)
    assert count == 1


def test_calibration_retrieves_similar_meal_and_is_bounded():
    memory = SimilarityMemory(k=1, capacity=2, strength=1)
    prediction = np.zeros(24)
    memory.observe(np.array([1., 0.]), prediction, np.ones(24) * 10, 1)
    memory.observe(np.array([0., 1.]), prediction, np.ones(24) * 30, 2)
    result, _ = memory.correct(np.array([0., 1.]), prediction, 3)
    np.testing.assert_allclose(result, 30)
    memory.observe(np.array([1., 0.]), prediction, np.ones(24) * 50, 3)
    assert len(memory.entries) == 2


def test_peak_metric_uses_curve_peak_not_endpoint():
    result = metrics([[100, 180, 120]], [[100, 170, 120]])
    assert result["peak_mae_mg_dl"] == 10
    assert result["peak_within_20_mg_dl_pct"] == 100


def test_multimodal_gradients_reach_both_branches():
    torch.manual_seed(42)
    model = MealLSTM()
    history = torch.randn(4, 12, 1, requires_grad=True)
    embedding = torch.randn(4, 1280, requires_grad=True)
    output = model(history, embedding)
    assert output.shape == (4, 24)
    output.square().mean().backward()
    assert history.grad.abs().sum() > 0
    assert embedding.grad.abs().sum() > 0
