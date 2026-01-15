from pathlib import Path

from src.dataset.annotations import BoxAnnotation, ImageAnnotation, load_annotation, save_annotation


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
