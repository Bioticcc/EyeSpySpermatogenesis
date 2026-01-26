from src.dataset.schema import LabelRecord, StageCluster, Quality, IncludeFlag, NegativeTrainingFlag


def test_stage_cluster_values() -> None:
    assert set(StageCluster.labeled_values()) == {
        StageCluster.XII_I,
        StageCluster.II_IV,
        StageCluster.V_VI,
        StageCluster.VII_VIII,
        StageCluster.IX_X,
        StageCluster.XI,
    }


def test_label_record_roundtrip() -> None:
    rec = LabelRecord(
        image_id="img1",
        rel_path="a/b.tif",
        stage_cluster=StageCluster.XI,
        quality=Quality.HIGH,
        include=IncludeFlag.YES,
        negative_training=NegativeTrainingFlag.YES,
        notes="note",
        labeled_by="tester",
        labeled_at="now",
    )
    row = rec.as_row()
    loaded = LabelRecord.from_row(row)
    assert loaded.image_id == rec.image_id
    assert loaded.stage_cluster == rec.stage_cluster
    assert loaded.quality == rec.quality
    assert loaded.include == rec.include
    assert loaded.negative_training == rec.negative_training
