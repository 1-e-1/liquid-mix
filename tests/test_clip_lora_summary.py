import json

import pandas as pd

from scripts.summarize_clip_lora_seeds import load_result


def test_load_result_computes_lora_deltas(tmp_path):
    out_dir = tmp_path / "q1_clip_lora_demo_seed7"
    out_dir.mkdir()
    (out_dir / "metadata.json").write_text(
        json.dumps({"dataset": "demo-dataset", "seed": 7}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": "clip_zero_shot_openai",
                "trainable_params": 0,
                "total_params": 100,
                "epochs": 0,
                "test_i2t_r@1": 0.1,
                "test_i2t_r@5": 0.2,
                "test_i2t_r@10": 0.3,
                "test_t2i_r@1": 0.4,
                "test_t2i_r@5": 0.5,
                "test_t2i_r@10": 0.6,
            },
            {
                "model": "clip_lora_finetuned",
                "trainable_params": 8,
                "total_params": 100,
                "epochs": 2,
                "test_i2t_r@1": 0.15,
                "test_i2t_r@5": 0.25,
                "test_i2t_r@10": 0.35,
                "test_t2i_r@1": 0.45,
                "test_t2i_r@5": 0.55,
                "test_t2i_r@10": 0.65,
            },
        ]
    ).to_csv(out_dir / "clip_lora_summary.csv", index=False)

    rows = load_result(out_dir)

    assert len(rows) == 6
    assert rows[0]["dataset"] == "demo-dataset"
    assert rows[0]["seed"] == 7
    assert rows[0]["metric"] == "i2t_r@1"
    assert rows[0]["delta"] == 0.04999999999999999
    assert rows[-1]["metric"] == "t2i_r@10"
