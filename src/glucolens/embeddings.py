from pathlib import Path
import json
import hashlib
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def encoder():
    torch.hub.set_dir(str(Path(__file__).resolve().parents[2] / "artifacts" / "torch-cache"))
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model.eval()
    return model, EfficientNet_B0_Weights.DEFAULT.transforms()


def embed_meals(prepared, batch_size=24):
    started = time.perf_counter()
    prepared = Path(prepared)
    meals = pd.read_csv(prepared / "meals.csv")
    model, transform = encoder()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(meals), batch_size):
            images = []
            for path in meals.image.iloc[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            chunks.append(model(torch.stack(images)).numpy())
            if start % (batch_size * 10) == 0:
                print(f"Embedded {min(start + batch_size, len(meals))}/{len(meals)} meals", flush=True)
    np.save(prepared / "embeddings.npy", np.concatenate(chunks))
    signature = hashlib.sha256((prepared / "meals.csv").read_bytes()).hexdigest()
    (prepared / "embeddings.json").write_text(json.dumps({"meal_manifest_sha256": signature,
        "backbone": "EfficientNet_B0_Weights.IMAGENET1K_V1", "dimension": 1280,
        "embedding_seconds": time.perf_counter() - started}, indent=2))
