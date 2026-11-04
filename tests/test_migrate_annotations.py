import json
from pathlib import Path

from src.dataset.migrate_annotations import migrate_annotations


def test_migrate_adds_orientation(tmp_path: Path) -> None:
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    data = {
        "image_id": "img1",
        "rel_path": "img1.tif",
        "include": "unknown",
        "quality": "unknown",
        "tubules": [
            {
                "tubule_id": "img1__0",
                "polygon_xy": [[1, 2], [3, 2], [3, 4]],
                "notes": "",
            }
        ],
    }
    path = ann_dir / "img1.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    updated_files, updated_tubules = migrate_annotations(ann_dir, in_place=True)
    assert updated_files == 1
    assert updated_tubules == 1

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["tubules"][0]["orientation"] == "unknown"
    assert loaded["tubules"][0]["partial"] is False
