from pathlib import Path
import joblib
import numpy as np
import torch
from .model import MealLSTM


def predict_food(macros, starting_glucose, artifact="artifacts/fast", diabetes_flag=None):
    values = np.asarray([*macros, starting_glucose], dtype="float32")
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Provide four finite nutrient amounts and starting glucose")
    if (values[:4] < 0).any() or not 40 <= starting_glucose <= 400:
        raise ValueError("Nutrients must be nonnegative; glucose must be 40–400 mg/dL")
    model = joblib.load(Path(artifact) / "food_model.joblib")
    if model.n_features_in_ == 6:
        from .status import validate_flag
        if diabetes_flag is None:
            raise ValueError("This model requires a binary diabetes flag")
        values = np.r_[values, validate_flag(diabetes_flag)]
    elif diabetes_flag is not None:
        raise ValueError("This checkpoint does not use diabetes status; load artifacts/status")
    return model.predict(values[None, :])[0] + starting_glucose


def predict_image(image, history, artifact="artifacts/multimodal", diabetes_flag=None):
    from .embeddings import encoder
    history = np.asarray(history, dtype="float32")
    if history.shape != (12,) or not np.isfinite(history).all() or ((history < 40) | (history > 400)).any():
        raise ValueError("Provide 12 glucose readings from -60 to -5 minutes, every 5 minutes")
    checkpoint = torch.load(Path(artifact) / "image_lstm.pt", map_location="cpu", weights_only=True)
    model = MealLSTM(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    context = None
    if model.context_dim:
        from .status import validate_flag
        if diabetes_flag is None:
            raise ValueError("This photo model requires a binary diabetes flag")
        context = torch.tensor([[validate_flag(diabetes_flag)]], dtype=torch.float32)
    elif diabetes_flag is not None:
        raise ValueError("This checkpoint does not use diabetes status; load artifacts/multimodal-status")
    cnn, transform = encoder()
    with torch.inference_mode():
        embedding = cnn(transform(image.convert("RGB")).unsqueeze(0))
        embedding = (embedding - checkpoint["embedding_mean"]) / checkpoint["embedding_std"]
        inputs = torch.tensor((history - 120) / 50).reshape(1, 12, 1)
        prediction = model(inputs, embedding, context).numpy()[0] * 50 + history[-1]
    return prediction
