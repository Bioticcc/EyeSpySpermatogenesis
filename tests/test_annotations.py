import json
from pathlib import Path

from src.dataset.annotations import (
    BoxAnnotation,
    ImageAnnotation,
    PolygonAnnotation,
    TubulePolygon,
    load_annotation,
    load_polygon_annotation,
    save_annotation,
    save_polygon_annotation,
)
from src.dataset.schema import IncludeFlag, Orientation, Quality


def test_annotation_roundtrip(tmp_path: Path) -> None:
    ann = ImageAnnotation(
        image_id="img1",
        rel_path="img1.tif",
        tissue="testis_tubule",
        boxes=[BoxAnnotation(box_id="b1", bbox_xywh=[1, 2, 3, 4], label="tubule", notes="")],
        labeled_by="tester",
        labeled_at="2020-01-01T00:00:00Z",
    )
    path = tmp_path / "img1.json"
    save_annotation(ann, path)
    loaded = load_annotation(path)
    assert loaded is not None
    assert loaded.image_id == ann.image_id
    assert loaded.rel_path == ann.rel_path
    assert loaded.boxes[0].bbox_xywh == ann.boxes[0].bbox_xywh


def test_polygon_annotation_roundtrip(tmp_path: Path) -> None:
    ann = PolygonAnnotation(
        image_id="img2",
        rel_path="img2.tif",
        include=IncludeFlag.YES,
        quality=Quality.HIGH,
        tubules=[
            TubulePolygon(
                tubule_id="t1",
                polygon_xy=[[1, 2], [3, 2], [3, 4]],
                orientation=Orientation.CROSS_SECTION,
                partial=True,
                notes="",
            )
        ],
        labeled_by="tester",
        labeled_at="2020-01-01T00:00:00Z",
    )
    path = tmp_path / "img2.json"
    save_polygon_annotation(ann, path)
    loaded = load_polygon_annotation(path)
    assert loaded is not None
    assert loaded.image_id == ann.image_id
    assert loaded.include == ann.include
    assert loaded.quality == ann.quality
    assert loaded.tubules[0].polygon_xy == ann.tubules[0].polygon_xy
    assert loaded.tubules[0].orientation == ann.tubules[0].orientation
    assert loaded.tubules[0].partial == ann.tubules[0].partial


def test_polygon_annotation_defaults_orientation(tmp_path: Path) -> None:
    data = {
        "image_id": "img3",
        "rel_path": "img3.tif",
        "include": "unknown",
        "quality": "unknown",
        "tubules": [
            {
                "tubule_id": "t1",
                "polygon_xy": [[1, 2], [3, 2], [3, 4]],
                "notes": "",
            }
        ],
    }
    path = tmp_path / "img3.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_polygon_annotation(path)
    assert loaded is not None
    assert loaded.tubules[0].orientation == Orientation.UNKNOWN
    assert loaded.tubules[0].partial is False


def test_polygon_annotation_rejects_invalid_orientation(tmp_path: Path) -> None:
    data = {
        "image_id": "img4",
        "rel_path": "img4.tif",
        "include": "unknown",
        "quality": "unknown",
        "tubules": [
            {
                "tubule_id": "t1",
                "polygon_xy": [[1, 2], [3, 2], [3, 4]],
                "orientation": "sideways",
                "notes": "",
            }
        ],
    }
    path = tmp_path / "img4.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    try:
        load_polygon_annotation(path)
    except ValueError as exc:
        assert "sideways" in str(exc)
        assert "t1" in str(exc)
    else:
        raise AssertionError("Expected invalid orientation to raise ValueError.")
