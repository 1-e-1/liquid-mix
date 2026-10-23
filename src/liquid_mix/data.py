from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
import shutil
from typing import Iterable, Sequence
from urllib.request import urlretrieve
from zipfile import ZipFile

from PIL import Image


@dataclass(frozen=True)
class CaptionSample:
    image_path: str
    caption: str
    source_id: str


def split_samples(
    samples: Sequence[CaptionSample],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[list[CaptionSample], list[CaptionSample], list[CaptionSample]]:
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val - n_test
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def save_jsonl(samples: Iterable[CaptionSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[CaptionSample]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            rows.append(CaptionSample(**payload))
    return rows


def caption_from_row(row: dict, preferred_key: str | None = None) -> str:
    def normalize(value) -> str:
        if isinstance(value, (list, tuple)):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
            return ""
        return str(value)

    if preferred_key and preferred_key in row:
        return normalize(row[preferred_key])
    if "text" in row:
        return normalize(row["text"])
    if "caption" in row:
        return normalize(row["caption"])
    caption_keys = sorted(k for k in row.keys() if k.startswith("caption_"))
    if caption_keys:
        return normalize(row[caption_keys[0]])
    raise KeyError(f"Could not find caption field in columns: {sorted(row.keys())}")


def materialize_hf_split(
    dataset_name: str,
    split: str,
    out_dir: str | Path,
    limit: int,
    caption_key: str | None = None,
    streaming: bool = True,
) -> list[CaptionSample]:
    from datasets import load_dataset

    out_dir = Path(out_dir)
    image_dir = out_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    samples: list[CaptionSample] = []
    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        image = row["image"].convert("RGB")
        image_path = image_dir / f"{idx:06d}.jpg"
        image.save(image_path, quality=92)
        samples.append(
            CaptionSample(
                image_path=str(image_path),
                caption=caption_from_row(row, caption_key),
                source_id=f"{dataset_name}:{split}:{idx}",
            )
        )
    return samples


def _image_members(zip_file: ZipFile) -> dict[str, str]:
    members = {}
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if name.startswith("._"):
            continue
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            members[name] = info.filename
    return members


def _extract_image(zip_file: ZipFile, member: str, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(member) as src:
        image = Image.open(src).convert("RGB")
        image.save(image_path, quality=92)


def materialize_flickr8k_archives(
    image_zip_path: str | Path,
    text_zip_path: str | Path,
    out_dir: str | Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
) -> tuple[list[CaptionSample], list[CaptionSample], list[CaptionSample]]:
    out_dir = Path(out_dir)
    with ZipFile(image_zip_path) as image_zip, ZipFile(text_zip_path) as text_zip:
        image_members = _image_members(image_zip)
        caption_lines = text_zip.read("Flickr8k.token.txt").decode("utf-8", errors="replace").splitlines()
        captions: dict[str, str] = {}
        for line in caption_lines:
            if not line.strip() or "\t" not in line:
                continue
            image_key, caption = line.split("\t", 1)
            image_name = image_key.split("#", 1)[0]
            captions.setdefault(image_name, caption.strip())

        def split_names(filename: str, limit: int) -> list[str]:
            names = [row.strip() for row in text_zip.read(filename).decode("utf-8", errors="replace").splitlines()]
            return [name for name in names if name][: int(limit)]

        split_to_names = {
            "train": split_names("Flickr_8k.trainImages.txt", train_limit),
            "val": split_names("Flickr_8k.devImages.txt", val_limit),
            "test": split_names("Flickr_8k.testImages.txt", test_limit),
        }
        outputs: dict[str, list[CaptionSample]] = {}
        for split, names in split_to_names.items():
            samples: list[CaptionSample] = []
            for idx, image_name in enumerate(names):
                if image_name not in image_members or image_name not in captions:
                    continue
                image_path = out_dir / "images" / split / image_name
                _extract_image(image_zip, image_members[image_name], image_path)
                samples.append(
                    CaptionSample(
                        image_path=str(image_path),
                        caption=captions[image_name],
                        source_id=f"flickr8k:{split}:{image_name}:{idx}",
                    )
                )
            outputs[split] = samples
    return outputs["train"], outputs["val"], outputs["test"]


def materialize_flickr30k_archive(
    archive_path: str | Path,
    out_dir: str | Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    seed: int = 42,
) -> tuple[list[CaptionSample], list[CaptionSample], list[CaptionSample]]:
    out_dir = Path(out_dir)
    with ZipFile(archive_path) as archive:
        image_members = _image_members(archive)
        with archive.open("flickr30k/captions.txt") as fh:
            rows = csv.DictReader((line.decode("utf-8", errors="replace") for line in fh))
            captions: dict[str, str] = {}
            for row in rows:
                image_name = str(row.get("image", "")).strip()
                caption = str(row.get("caption", "")).strip()
                if image_name and caption:
                    captions.setdefault(image_name, caption)

        all_samples: list[CaptionSample] = []
        for image_name, caption in captions.items():
            if image_name not in image_members:
                continue
            all_samples.append(
                CaptionSample(
                    image_path=image_name,
                    caption=caption,
                    source_id=f"flickr30k:{image_name}",
                )
            )
        total = int(train_limit) + int(val_limit) + int(test_limit)
        train, val, test = split_samples(
            all_samples,
            seed=seed,
            val_fraction=int(val_limit) / total,
            test_fraction=int(test_limit) / total,
        )
        split_to_samples = {
            "train": train[: int(train_limit)],
            "val": val[: int(val_limit)],
            "test": test[: int(test_limit)],
        }
        outputs: dict[str, list[CaptionSample]] = {}
        for split, samples in split_to_samples.items():
            materialized = []
            for idx, sample in enumerate(samples):
                image_name = Path(sample.image_path).name
                image_path = out_dir / "images" / split / image_name
                _extract_image(archive, image_members[image_name], image_path)
                materialized.append(
                    CaptionSample(
                        image_path=str(image_path),
                        caption=sample.caption,
                        source_id=f"flickr30k:{split}:{image_name}:{idx}",
                    )
                )
            outputs[split] = materialized
    return outputs["train"], outputs["val"], outputs["test"]


def _read_coco_captions(annotation_zip: ZipFile, member: str) -> list[CaptionSample]:
    payload = json.loads(annotation_zip.read(member).decode("utf-8"))
    image_by_id = {int(img["id"]): img["file_name"] for img in payload["images"]}
    first_caption: dict[int, str] = {}
    for ann in payload["annotations"]:
        image_id = int(ann["image_id"])
        caption = str(ann.get("caption", "")).strip()
        if caption and image_id not in first_caption:
            first_caption[image_id] = caption
    samples = []
    for image_id, file_name in image_by_id.items():
        caption = first_caption.get(image_id)
        if not caption:
            continue
        split = "train2014" if "train2014" in file_name else "val2014"
        samples.append(
            CaptionSample(
                image_path=file_name,
                caption=caption,
                source_id=f"coco2014:{split}:{image_id}",
            )
        )
    return samples


def _coco_file_name_to_member(image_members: dict[str, str], file_name: str, split: str) -> str | None:
    direct_key = Path(file_name).name
    if direct_key in image_members:
        member = image_members[direct_key]
        if f"/{split}/" in member.replace("\\", "/") or member.replace("\\", "/").endswith(f"{split}/{direct_key}"):
            return member
    # Fall back to any member with the same file name. This keeps the loader
    # compatible with repackaged COCO archives while still preferring split paths.
    return image_members.get(direct_key)


def _materialize_coco_samples(
    image_zip: ZipFile,
    image_members: dict[str, str],
    samples: Sequence[CaptionSample],
    out_dir: Path,
    split: str,
    limit: int,
) -> list[CaptionSample]:
    out = []
    for idx, sample in enumerate(samples[: int(limit)]):
        file_name = Path(sample.image_path).name
        member = _coco_file_name_to_member(image_members, file_name, "train2014" if "train" in split else "val2014")
        if member is None:
            continue
        image_path = out_dir / "images" / split / file_name
        _extract_image(image_zip, member, image_path)
        out.append(
            CaptionSample(
                image_path=str(image_path),
                caption=sample.caption,
                source_id=f"coco2014:{split}:{file_name}:{idx}",
            )
        )
    return out


def _copy_or_download_coco_image(
    image_zip: ZipFile,
    image_members: dict[str, str],
    file_name: str,
    out_path: Path,
    supplemental_image_dir: Path | None = None,
    download_missing: bool = False,
) -> bool:
    split = "train2014" if "train2014" in file_name else "val2014"
    member = _coco_file_name_to_member(image_members, file_name, split)
    if member is not None:
        _extract_image(image_zip, member, out_path)
        return True

    supplemental_path = None
    if supplemental_image_dir is not None:
        supplemental_path = supplemental_image_dir / split / file_name
        if supplemental_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(supplemental_path, out_path)
            return True

    if download_missing:
        if supplemental_path is None:
            raise ValueError("supplemental_image_dir is required when download_missing=True")
        supplemental_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"http://images.cocodataset.org/{split}/{file_name}"
        urlretrieve(url, supplemental_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(supplemental_path, out_path)
        return True
    return False


def _karpathy_caption(image: dict) -> str:
    sentences = image.get("sentences", [])
    for sentence in sentences:
        caption = str(sentence.get("raw", "")).strip()
        if caption:
            return caption
    raise ValueError(f"Karpathy image has no caption: {image.get('filename')}")


def _materialize_karpathy_samples(
    image_zip: ZipFile,
    image_members: dict[str, str],
    images: Sequence[dict],
    out_dir: Path,
    split: str,
    limit: int,
    supplemental_image_dir: Path | None = None,
    download_missing: bool = False,
) -> list[CaptionSample]:
    out = []
    for idx, image in enumerate(images[: int(limit)]):
        file_name = Path(str(image["filename"])).name
        image_path = out_dir / "images" / split / file_name
        if not _copy_or_download_coco_image(
            image_zip,
            image_members,
            file_name,
            image_path,
            supplemental_image_dir=supplemental_image_dir,
            download_missing=download_missing,
        ):
            continue
        out.append(
            CaptionSample(
                image_path=str(image_path),
                caption=_karpathy_caption(image),
                source_id=f"coco2014_karpathy:{split}:{file_name}:{idx}",
            )
        )
    return out


def materialize_coco2014_karpathy_split(
    image_zip_path: str | Path,
    split_json_path: str | Path,
    out_dir: str | Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    supplemental_image_dir: str | Path | None = None,
    download_missing: bool = False,
) -> tuple[list[CaptionSample], list[CaptionSample], list[CaptionSample]]:
    """Materialize COCO 2014 samples according to the Karpathy caption split."""

    out_dir = Path(out_dir)
    payload = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    images = payload["images"]
    train_images = [img for img in images if img.get("split") in {"train", "restval"}]
    val_images = [img for img in images if img.get("split") == "val"]
    test_images = [img for img in images if img.get("split") == "test"]
    supplemental_dir = Path(supplemental_image_dir) if supplemental_image_dir is not None else None

    with ZipFile(image_zip_path) as image_zip:
        image_members = _image_members(image_zip)
        train = _materialize_karpathy_samples(
            image_zip,
            image_members,
            train_images,
            out_dir,
            "train",
            train_limit,
            supplemental_image_dir=supplemental_dir,
            download_missing=download_missing,
        )
        val = _materialize_karpathy_samples(
            image_zip,
            image_members,
            val_images,
            out_dir,
            "val",
            val_limit,
            supplemental_image_dir=supplemental_dir,
            download_missing=download_missing,
        )
        test = _materialize_karpathy_samples(
            image_zip,
            image_members,
            test_images,
            out_dir,
            "test",
            test_limit,
            supplemental_image_dir=supplemental_dir,
            download_missing=download_missing,
        )
    return train, val, test


def materialize_coco2014_archives(
    image_zip_path: str | Path,
    annotation_zip_path: str | Path,
    out_dir: str | Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    seed: int = 42,
    use_5k_part: bool = True,
) -> tuple[list[CaptionSample], list[CaptionSample], list[CaptionSample]]:
    """Materialize COCO 2014 train/val/test jsonl-compatible samples.

    The local dataset supplied for this project is a repackaged COCO archive:
    one zip contains train/val images and a `coco2014/5k.part` list, while a
    second zip contains the official caption annotations. This loader supports
    that layout and also works with archives that preserve official file names.
    For retrieval, the test split is taken from `5k.part` when available.
    """

    out_dir = Path(out_dir)
    with ZipFile(image_zip_path) as image_zip, ZipFile(annotation_zip_path) as ann_zip:
        image_members = _image_members(image_zip)
        train_samples = _read_coco_captions(
            ann_zip,
            "annotations_trainval2014/annotations/captions_train2014.json",
        )
        val_samples = _read_coco_captions(
            ann_zip,
            "annotations_trainval2014/annotations/captions_val2014.json",
        )

        train_samples = sorted(train_samples, key=lambda s: s.image_path)
        val_by_name = {Path(sample.image_path).name: sample for sample in val_samples}
        if use_5k_part:
            try:
                part_lines = image_zip.read("coco2014/5k.part").decode("utf-8", errors="replace").splitlines()
                ordered_names = [Path(line.strip()).name for line in part_lines if line.strip()]
                test_candidates = [val_by_name[name] for name in ordered_names if name in val_by_name]
            except KeyError:
                test_candidates = []
        else:
            test_candidates = []
        if not test_candidates:
            rng = random.Random(seed)
            test_candidates = list(val_samples)
            rng.shuffle(test_candidates)

        test_names = {Path(sample.image_path).name for sample in test_candidates[: int(test_limit)]}
        remaining_val = [sample for sample in sorted(val_samples, key=lambda s: s.image_path) if Path(sample.image_path).name not in test_names]
        if len(remaining_val) < int(val_limit):
            remaining_val = [sample for sample in sorted(val_samples, key=lambda s: s.image_path) if Path(sample.image_path).name not in test_names]

        train = _materialize_coco_samples(image_zip, image_members, train_samples, out_dir, "train", train_limit)
        val = _materialize_coco_samples(image_zip, image_members, remaining_val, out_dir, "val", val_limit)
        test = _materialize_coco_samples(image_zip, image_members, test_candidates, out_dir, "test", test_limit)
    return train, val, test
