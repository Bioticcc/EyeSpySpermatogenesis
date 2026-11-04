from pathlib import Path

import numpy as np
import torch
import tifffile

from src.dataset.annotations import PolygonAnnotation, TubulePolygon, save_polygon_annotation
from src.dataset.seg_dataset import TubuleSegmentationDataset
from src.dataset.schema import IncludeFlag, LabelRecord, Orientation, Quality


def test_segmentation_dataset_returns_mask(tmp_path: Path) -> None:
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
    ann = PolygonAnnotation(
        image_id="sample",
        rel_path="sample.tif",
        include=IncludeFlag.YES,
        quality=Quality.HIGH,
        tubules=[
            TubulePolygon(
                tubule_id="sample__0",
                polygon_xy=[[10, 10], [40, 10], [40, 40], [10, 40]],
                orientation=Orientation.CROSS_SECTION,
            )
        ],
        labeled_by="tester",
        labeled_at="2020-01-01T00:00:00Z",
    )
    save_polygon_annotation(ann, ann_dir / "sample.json")

    dataset = TubuleSegmentationDataset(
        [record],
        raw_dir=raw_dir,
        annotations_dir=ann_dir,
        max_size=32,
        augment=False,
        require_annotations=True,
        instance_mode=False,
    )
    image, mask, neg_mask = dataset[0]
    assert image.shape[0] == 3
    assert mask.shape[0] == 1
    assert neg_mask.shape[0] == 1
    assert mask.dtype == torch.float32
    assert mask.sum() > 0
    assert neg_mask.sum() == 0
