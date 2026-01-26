from pathlib import Path
import numpy as np
import tifffile
import torch

from src.app.inference import load_segmenter, predict_segmentation
from src.training.seg_model import create_segmentation_model


def test_inference_roundtrip(tmp_path: Path) -> None:
    # create synthetic image
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "sample.tif"
    tifffile.imwrite(img_path, arr)

    # create dummy weights
    model = create_segmentation_model()
    weights_path = tmp_path / "seg_model.pt"
    torch.save(model.state_dict(), weights_path)

    loaded = load_segmenter(weights_path)
    result = predict_segmentation(img_path, loaded, threshold=0.5, max_size=64)
    assert "detections" in result
    assert isinstance(result["detections"], list)
    assert result["mask"].shape[:2] == (32, 32)
