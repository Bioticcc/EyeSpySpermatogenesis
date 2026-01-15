from pathlib import Path
import numpy as np
import tifffile
import torch

from src.app.inference import load_detector, predict_image
from src.training.detect_model import create_detector


def test_inference_roundtrip(tmp_path: Path) -> None:
    # create synthetic image
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "sample.tif"
    tifffile.imwrite(img_path, arr)

    # create dummy weights
    model = create_detector()
    weights_path = tmp_path / "detector.pt"
    torch.save(model.state_dict(), weights_path)

    loaded = load_detector(weights_path)
    result = predict_image(img_path, loaded, score_threshold=0.0, nms_iou=0.0, max_size=64)
    assert "detections" in result
    assert isinstance(result["detections"], list)
