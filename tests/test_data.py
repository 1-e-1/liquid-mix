from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from liquid_mix.data import (
    CaptionSample,
    caption_from_row,
    load_jsonl,
    materialize_coco2014_karpathy_split,
    materialize_flickr8k_archives,
    materialize_flickr30k_archive,
    save_jsonl,
    split_samples,
)


def _make_samples(tmp_path: Path, n: int = 4):
    samples = []
    for idx in range(n):
        image_path = tmp_path / f"sample_{idx}.jpg"
        Image.new("RGB", (16, 16), color=(idx * 40, 20, 100)).save(image_path)
        samples.append(CaptionSample(image_path=str(image_path), caption=f"caption {idx}", source_id=str(idx)))
    return samples


def test_split_samples_is_deterministic(tmp_path):
    samples = _make_samples(tmp_path, 10)

    first = split_samples(samples, seed=7, val_fraction=0.2, test_fraction=0.2)
    second = split_samples(samples, seed=7, val_fraction=0.2, test_fraction=0.2)

    assert [[s.source_id for s in part] for part in first] == [[s.source_id for s in part] for part in second]
    assert [len(part) for part in first] == [6, 2, 2]


def test_jsonl_round_trip_preserves_samples(tmp_path):
    samples = _make_samples(tmp_path, 4)
    path = tmp_path / "samples.jsonl"

    save_jsonl(samples, path)
    loaded = load_jsonl(path)

    assert loaded == samples


def test_caption_from_row_prefers_text_then_caption_columns():
    assert caption_from_row({"text": "a caption"}) == "a caption"
    assert caption_from_row({"caption": "fallback"}) == "fallback"
    assert caption_from_row({"caption_1": "second", "caption_0": "first"}) == "first"


def test_caption_from_row_uses_first_item_when_caption_is_list():
    assert caption_from_row({"caption": ["first caption", "second caption"]}) == "first caption"


def _image_bytes(color=(20, 40, 80)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_materialize_flickr8k_archives_uses_official_splits(tmp_path):
    image_zip = tmp_path / "images.zip"
    text_zip = tmp_path / "text.zip"
    with ZipFile(image_zip, "w") as zf:
        for name in ["train.jpg", "dev.jpg", "test.jpg"]:
            zf.writestr(f"Flicker8k_Dataset/{name}", _image_bytes())
    with ZipFile(text_zip, "w") as zf:
        zf.writestr("Flickr8k.token.txt", "train.jpg#0\ttrain caption\ndev.jpg#0\tdev caption\ntest.jpg#0\ttest caption\n")
        zf.writestr("Flickr_8k.trainImages.txt", "train.jpg\n")
        zf.writestr("Flickr_8k.devImages.txt", "dev.jpg\n")
        zf.writestr("Flickr_8k.testImages.txt", "test.jpg\n")

    train, val, test = materialize_flickr8k_archives(image_zip, text_zip, tmp_path / "out", 1, 1, 1)

    assert [sample.caption for sample in train] == ["train caption"]
    assert [sample.caption for sample in val] == ["dev caption"]
    assert [sample.caption for sample in test] == ["test caption"]
    assert Path(train[0].image_path).exists()


def test_materialize_flickr30k_archive_reads_captions_csv(tmp_path):
    archive = tmp_path / "flickr30k.zip"
    with ZipFile(archive, "w") as zf:
        for name in ["a.jpg", "b.jpg", "c.jpg"]:
            zf.writestr(f"flickr30k/Images/{name}", _image_bytes())
        zf.writestr(
            "flickr30k/captions.txt",
            "image,caption\n"
            "a.jpg,first a\n"
            "a.jpg,second a\n"
            "b.jpg,first b\n"
            "c.jpg,first c\n",
        )

    train, val, test = materialize_flickr30k_archive(archive, tmp_path / "out", 1, 1, 1, seed=3)

    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
    assert {sample.caption for sample in [*train, *val, *test]} == {"first a", "first b", "first c"}
    assert all(Path(sample.image_path).exists() for sample in [*train, *val, *test])


def test_materialize_coco_karpathy_uses_supplemental_images(tmp_path):
    image_zip = tmp_path / "coco_images.zip"
    with ZipFile(image_zip, "w") as zf:
        zf.writestr("train2014/COCO_train2014_000000000001.jpg", _image_bytes())
        zf.writestr("val2014/COCO_val2014_000000000002.jpg", _image_bytes())
    supplemental = tmp_path / "supplemental" / "val2014"
    supplemental.mkdir(parents=True)
    (supplemental / "COCO_val2014_000000000003.jpg").write_bytes(_image_bytes(color=(80, 20, 40)))
    split_json = tmp_path / "dataset_coco_karpathy.json"
    split_json.write_text(
        """
{
  "dataset": "coco",
  "images": [
    {
      "filename": "COCO_train2014_000000000001.jpg",
      "split": "train",
      "sentences": [{"raw": "train caption"}]
    },
    {
      "filename": "COCO_val2014_000000000002.jpg",
      "split": "val",
      "sentences": [{"raw": "val caption"}]
    },
    {
      "filename": "COCO_val2014_000000000003.jpg",
      "split": "test",
      "sentences": [{"raw": "test caption"}]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    train, val, test = materialize_coco2014_karpathy_split(
        image_zip,
        split_json,
        tmp_path / "out",
        train_limit=1,
        val_limit=1,
        test_limit=1,
        supplemental_image_dir=tmp_path / "supplemental",
    )

    assert [sample.caption for sample in train] == ["train caption"]
    assert [sample.caption for sample in val] == ["val caption"]
    assert [sample.caption for sample in test] == ["test caption"]
    assert all(Path(sample.image_path).exists() for sample in [*train, *val, *test])
