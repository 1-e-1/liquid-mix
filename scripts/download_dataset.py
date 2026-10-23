from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from liquid_mix.data import (
    load_jsonl,
    materialize_coco2014_karpathy_split,
    materialize_coco2014_archives,
    materialize_flickr8k_archives,
    materialize_flickr30k_archive,
    materialize_hf_split,
    save_jsonl,
    split_samples,
)


def prepare_dataset(config: dict, force: bool = False) -> tuple[Path, int, int, int]:
    dataset_cfg = config["dataset"]
    root = ROOT / dataset_cfg["processed_dir"]
    train_path = root / "train.jsonl"
    val_path = root / "val.jsonl"
    test_path = root / "test.jsonl"
    if not force and train_path.exists() and val_path.exists() and test_path.exists():
        n_train = len(load_jsonl(train_path))
        n_val = len(load_jsonl(val_path))
        n_test = len(load_jsonl(test_path))
        if (
            n_train == int(dataset_cfg["train_limit"])
            and n_val == int(dataset_cfg["val_limit"])
            and n_test == int(dataset_cfg["test_limit"])
        ):
            return root, n_train, n_val, n_test

    root.mkdir(parents=True, exist_ok=True)
    name = dataset_cfg["name"]
    seed = int(dataset_cfg.get("seed", 42))
    train_limit = int(dataset_cfg["train_limit"])
    val_limit = int(dataset_cfg["val_limit"])
    test_limit = int(dataset_cfg["test_limit"])
    try:
        source = dataset_cfg.get("source", "hf")
        if source == "flickr8k_zip":
            train, val, test = materialize_flickr8k_archives(
                ROOT / dataset_cfg["image_zip"],
                ROOT / dataset_cfg["text_zip"],
                root,
                train_limit,
                val_limit,
                test_limit,
            )
        elif source == "flickr30k_zip":
            train, val, test = materialize_flickr30k_archive(
                ROOT / dataset_cfg["archive"],
                root,
                train_limit,
                val_limit,
                test_limit,
                seed=seed,
            )
        elif source == "coco2014_zip":
            train, val, test = materialize_coco2014_archives(
                ROOT / dataset_cfg["image_zip"],
                ROOT / dataset_cfg["annotation_zip"],
                root,
                train_limit,
                val_limit,
                test_limit,
                seed=seed,
                use_5k_part=bool(dataset_cfg.get("use_5k_part", True)),
            )
        elif source == "coco2014_karpathy_json":
            train, val, test = materialize_coco2014_karpathy_split(
                ROOT / dataset_cfg["image_zip"],
                ROOT / dataset_cfg["split_json"],
                root,
                train_limit,
                val_limit,
                test_limit,
                supplemental_image_dir=ROOT / dataset_cfg["supplemental_image_dir"]
                if dataset_cfg.get("supplemental_image_dir")
                else None,
                download_missing=bool(dataset_cfg.get("download_missing_images", False)),
            )
        elif name == "reach-vb/pokemon-blip-captions":
            total = train_limit + val_limit + test_limit
            all_samples = materialize_hf_split(name, "train", root, total, streaming=False)
            train, val, test = split_samples(
                all_samples,
                seed=seed,
                val_fraction=val_limit / total,
                test_fraction=test_limit / total,
            )
        else:
            train = materialize_hf_split(name, "train", root, train_limit, streaming=True)
            val = materialize_hf_split(name, "validation", root, val_limit, streaming=True)
            test = materialize_hf_split(name, "test", root, test_limit, streaming=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare dataset {name}: {type(exc).__name__}: {exc}") from exc

    train = train[:train_limit]
    val = val[:val_limit]
    test = test[:test_limit]
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    save_jsonl(test, test_path)
    return root, len(train), len(val), len(test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clip_vlm_pokemon.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    root, n_train, n_val, n_test = prepare_dataset(config, force=args.force)
    print(f"dataset_dir={root}")
    print(f"train={n_train} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()
