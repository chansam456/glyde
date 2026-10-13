from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimilarityMemory:
    """Offline per-person residual calibrator with timestamp-gated observations."""
    k: int = 5
    capacity: int = 40
    strength: float = 0.5
    entries: list = field(default_factory=list)

    def correct(self, embedding, prediction, now):
        available = [x for x in self.entries if x[0] <= now]
        if not available:
            return prediction.copy(), 0
        unit = embedding / max(np.linalg.norm(embedding), 1e-8)
        scores = np.array([np.dot(unit, x[1]) for x in available])
        indices = np.argsort(scores)[-self.k:]
        weights = np.exp((scores[indices] - scores[indices].max()) / 0.1)
        weights /= weights.sum()
        residual = sum(w * available[i][2] for w, i in zip(weights, indices))
        # Shrink early updates: a single observed meal should not dominate a forecast.
        gain = self.strength * min(len(available) / self.k, 1)
        return prediction + gain * residual, len(available)

    def observe(self, embedding, prediction, target, available_at):
        unit = embedding / max(np.linalg.norm(embedding), 1e-8)
        self.entries.append((available_at, unit.copy(), (target - prediction).copy()))
        self.entries = self.entries[-self.capacity:]
