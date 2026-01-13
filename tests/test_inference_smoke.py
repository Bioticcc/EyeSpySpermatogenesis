from pathlib import Path
import numpy as np
import tifffile
import torch

from src.app.inference import load_model, predict_image
from src.training.model import create_model


def test_inference_roundtrip(tmp_path: Path) -> None:
    # create synthetic image
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "sample.tif"
    tifffile.imwrite(img_path, arr)

    # create dummy weights
    model = create_model()
    weights_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), weights_path)

    loaded = load_model(weights_path)
    result = predict_image(img_path, loaded)
    assert "predicted" in result
    assert len(result["top3"]) == 3
