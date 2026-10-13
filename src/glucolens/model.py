import torch
from torch import nn


class MealLSTM(nn.Module):
    """Fuse a frozen EfficientNet embedding with strictly premeal CGM history."""

    def __init__(self, image_dim=1280, hidden=48, horizon=24, layers=1, context_dim=0):
        super().__init__()
        self.context_dim = context_dim
        self.history = nn.LSTM(1, hidden, batch_first=True, num_layers=layers,
                               dropout=.15 if layers > 1 else 0.)
        self.image = nn.Sequential(nn.Linear(image_dim, 64), nn.ReLU(), nn.Dropout(0.15))
        self.head = nn.Sequential(nn.Linear(hidden + 64 + context_dim, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, history, embedding, context=None):
        _, (hidden, _) = self.history(history)
        features = [hidden[-1], self.image(embedding)]
        if self.context_dim:
            if context is None or context.shape != (history.shape[0], self.context_dim):
                raise ValueError("Binary diabetes context is required for this checkpoint")
            features.append(context)
        return self.head(torch.cat(features, dim=1))
