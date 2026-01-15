from pathlib import Path

import numpy as np
import torch
import tifffile

from src.dataset.annotations import BoxAnnotation, ImageAnnotation, save_annotation
from src.dataset.detect_dataset import TubuleDetectDataset
from src.dataset.schema import IncludeFlag, LabelRecord, Quality


def test_detect_dataset_returns_scaled_boxes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    ann_dir = tmp_path / "annotations"
    raw_dir.mkdir()
    ann_dir.mkdir()

    arr = (np.random.rand(64, 64) * 65535).astype(np.uint16)
    img_path = raw_dir / "sample.tif"
    tifffile.imwrite(img_path, arr)

    record = LabelRecord(
        image_id="sample",
        rel_path="sample.tif",
        quality=Quality.HIGH,
        include=IncludeFlag.YES,
    )
    ann = ImageAnnotation(
        image_id="sample",
        rel_path="sample.tif",
        boxes=[BoxAnnotation(box_id="b1", bbox_xywh=[10, 10, 20, 20])],
        labeled_by="tester",
        labeled_at="2020-01-01T00:00:00Z",
    )
    save_annotation(ann, ann_dir / "sample.json")

    dataset = TubuleDetectDataset(
        [record],
        raw_dir=raw_dir,
        annotations_dir=ann_dir,
        max_size=32,
        augment=False,
        require_annotations=True,
    )
    image, target = dataset[0]
    assert image.shape[0] == 3
    assert target["boxes"].shape == (1, 4)
    expected = torch.tensor([5.0, 5.0, 15.0, 15.0])
    assert torch.allclose(target["boxes"][0], expected, atol=1e-4)
