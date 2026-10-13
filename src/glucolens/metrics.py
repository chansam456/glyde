import numpy as np


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    peak_error = prediction.max(axis=1) - target.max(axis=1)
    return {
        "trajectory_mae_mg_dl": float(np.abs(prediction - target).mean()),
        "trajectory_rmse_mg_dl": float(np.sqrt(((prediction - target) ** 2).mean())),
        "peak_mae_mg_dl": float(np.abs(peak_error).mean()),
        "peak_within_20_mg_dl_pct": float((np.abs(peak_error) <= 20).mean() * 100),
        "peak_time_mae_minutes": float(np.abs(prediction.argmax(1) - target.argmax(1)).mean() * 5),
        "meals": len(target),
    }


def participant_bootstrap(frame, repeats=1000, seed=42):
    """Paired bootstrap over people, preserving within-person dependence."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby("participant")]
    estimates = []
    for _ in range(repeats):
        sampled = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        base = np.concatenate([g.base_peak_error.to_numpy() for g in sampled]).mean()
        personal = np.concatenate([g.personal_peak_error.to_numpy() for g in sampled]).mean()
        estimates.append(100 * (base - personal) / max(base, 1e-8))
    return np.percentile(estimates, [2.5, 97.5]).tolist()
