# CLIP-LoRA Multi-seed Baseline Summary

This report aggregates matched PEFT baselines trained with the same lightweight two-epoch LoRA budget.
Deltas are LoRA minus zero-shot CLIP on the same test split.

Positive per-seed metric deltas: `54/54`.

## Mean and Range

| dataset | seeds | metric | clip_mean | lora_mean | delta_mean | delta_std | delta_min | delta_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coco2014-1k | 3 | i2t_r@1 | 0.5380 | 0.5490 | 0.0110 | 0.0010 | 0.0100 | 0.0120 |
| coco2014-1k | 3 | i2t_r@5 | 0.8140 | 0.8217 | 0.0077 | 0.0021 | 0.0060 | 0.0100 |
| coco2014-1k | 3 | i2t_r@10 | 0.9000 | 0.9027 | 0.0027 | 0.0012 | 0.0020 | 0.0040 |
| coco2014-1k | 3 | t2i_r@1 | 0.4900 | 0.5007 | 0.0107 | 0.0006 | 0.0100 | 0.0110 |
| coco2014-1k | 3 | t2i_r@5 | 0.7820 | 0.7873 | 0.0053 | 0.0012 | 0.0040 | 0.0060 |
| coco2014-1k | 3 | t2i_r@10 | 0.8810 | 0.8857 | 0.0047 | 0.0015 | 0.0030 | 0.0060 |
| flickr30k-large-local | 3 | i2t_r@1 | 0.7140 | 0.7430 | 0.0290 | 0.0026 | 0.0270 | 0.0320 |
| flickr30k-large-local | 3 | i2t_r@5 | 0.9050 | 0.9303 | 0.0253 | 0.0012 | 0.0240 | 0.0260 |
| flickr30k-large-local | 3 | i2t_r@10 | 0.9420 | 0.9607 | 0.0187 | 0.0006 | 0.0180 | 0.0190 |
| flickr30k-large-local | 3 | t2i_r@1 | 0.6840 | 0.7157 | 0.0317 | 0.0025 | 0.0290 | 0.0340 |
| flickr30k-large-local | 3 | t2i_r@5 | 0.8950 | 0.9190 | 0.0240 | 0.0000 | 0.0240 | 0.0240 |
| flickr30k-large-local | 3 | t2i_r@10 | 0.9430 | 0.9550 | 0.0120 | 0.0010 | 0.0110 | 0.0130 |
| flickr8k-full-local | 3 | i2t_r@1 | 0.5760 | 0.6060 | 0.0300 | 0.0026 | 0.0280 | 0.0330 |
| flickr8k-full-local | 3 | i2t_r@5 | 0.8350 | 0.8620 | 0.0270 | 0.0026 | 0.0240 | 0.0290 |
| flickr8k-full-local | 3 | i2t_r@10 | 0.9140 | 0.9330 | 0.0190 | 0.0020 | 0.0170 | 0.0210 |
| flickr8k-full-local | 3 | t2i_r@1 | 0.5450 | 0.5843 | 0.0393 | 0.0025 | 0.0370 | 0.0420 |
| flickr8k-full-local | 3 | t2i_r@5 | 0.8180 | 0.8477 | 0.0297 | 0.0021 | 0.0280 | 0.0320 |
| flickr8k-full-local | 3 | t2i_r@10 | 0.9020 | 0.9120 | 0.0100 | 0.0017 | 0.0080 | 0.0110 |

## Per-seed Results

| dataset | seed | metric | clip | lora | delta | trainable_params | total_params | epochs | source_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coco2014-1k | 7 | i2t_r@1 | 0.5380 | 0.5500 | 0.0120 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed7 |
| coco2014-1k | 7 | i2t_r@5 | 0.8140 | 0.8210 | 0.0070 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed7 |
| coco2014-1k | 7 | i2t_r@10 | 0.9000 | 0.9020 | 0.0020 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed7 |
| coco2014-1k | 7 | t2i_r@1 | 0.4900 | 0.5000 | 0.0100 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed7 |
| coco2014-1k | 7 | t2i_r@5 | 0.7820 | 0.7880 | 0.0060 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed7 |
| coco2014-1k | 7 | t2i_r@10 | 0.8810 | 0.8870 | 0.0060 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed7 |
| coco2014-1k | 13 | i2t_r@1 | 0.5380 | 0.5490 | 0.0110 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed13 |
| coco2014-1k | 13 | i2t_r@5 | 0.8140 | 0.8240 | 0.0100 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed13 |
| coco2014-1k | 13 | i2t_r@10 | 0.9000 | 0.9020 | 0.0020 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed13 |
| coco2014-1k | 13 | t2i_r@1 | 0.4900 | 0.5010 | 0.0110 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed13 |
| coco2014-1k | 13 | t2i_r@5 | 0.7820 | 0.7860 | 0.0040 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed13 |
| coco2014-1k | 13 | t2i_r@10 | 0.8810 | 0.8860 | 0.0050 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed13 |
| coco2014-1k | 21 | i2t_r@1 | 0.5380 | 0.5480 | 0.0100 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed21 |
| coco2014-1k | 21 | i2t_r@5 | 0.8140 | 0.8200 | 0.0060 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed21 |
| coco2014-1k | 21 | i2t_r@10 | 0.9000 | 0.9040 | 0.0040 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed21 |
| coco2014-1k | 21 | t2i_r@1 | 0.4900 | 0.5010 | 0.0110 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed21 |
| coco2014-1k | 21 | t2i_r@5 | 0.7820 | 0.7880 | 0.0060 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed21 |
| coco2014-1k | 21 | t2i_r@10 | 0.8810 | 0.8840 | 0.0030 | 245760 | 151523073 | 2 | results\q1_clip_lora_coco1k_seed21 |
| flickr30k-large-local | 7 | i2t_r@1 | 0.7140 | 0.7410 | 0.0270 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed7 |
| flickr30k-large-local | 7 | i2t_r@5 | 0.9050 | 0.9310 | 0.0260 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed7 |
| flickr30k-large-local | 7 | i2t_r@10 | 0.9420 | 0.9600 | 0.0180 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed7 |
| flickr30k-large-local | 7 | t2i_r@1 | 0.6840 | 0.7160 | 0.0320 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed7 |
| flickr30k-large-local | 7 | t2i_r@5 | 0.8950 | 0.9190 | 0.0240 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed7 |
| flickr30k-large-local | 7 | t2i_r@10 | 0.9430 | 0.9540 | 0.0110 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed7 |
| flickr30k-large-local | 13 | i2t_r@1 | 0.7140 | 0.7420 | 0.0280 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed13 |
| flickr30k-large-local | 13 | i2t_r@5 | 0.9050 | 0.9290 | 0.0240 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed13 |
| flickr30k-large-local | 13 | i2t_r@10 | 0.9420 | 0.9610 | 0.0190 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed13 |
| flickr30k-large-local | 13 | t2i_r@1 | 0.6840 | 0.7180 | 0.0340 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed13 |
| flickr30k-large-local | 13 | t2i_r@5 | 0.8950 | 0.9190 | 0.0240 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed13 |
| flickr30k-large-local | 13 | t2i_r@10 | 0.9430 | 0.9550 | 0.0120 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed13 |
| flickr30k-large-local | 21 | i2t_r@1 | 0.7140 | 0.7460 | 0.0320 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed21 |
| flickr30k-large-local | 21 | i2t_r@5 | 0.9050 | 0.9310 | 0.0260 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed21 |
| flickr30k-large-local | 21 | i2t_r@10 | 0.9420 | 0.9610 | 0.0190 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed21 |
| flickr30k-large-local | 21 | t2i_r@1 | 0.6840 | 0.7130 | 0.0290 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed21 |
| flickr30k-large-local | 21 | t2i_r@5 | 0.8950 | 0.9190 | 0.0240 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed21 |
| flickr30k-large-local | 21 | t2i_r@10 | 0.9430 | 0.9560 | 0.0130 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr30k_large_seed21 |
| flickr8k-full-local | 7 | i2t_r@1 | 0.5760 | 0.6040 | 0.0280 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed7 |
| flickr8k-full-local | 7 | i2t_r@5 | 0.8350 | 0.8590 | 0.0240 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed7 |
| flickr8k-full-local | 7 | i2t_r@10 | 0.9140 | 0.9350 | 0.0210 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed7 |
| flickr8k-full-local | 7 | t2i_r@1 | 0.5450 | 0.5870 | 0.0420 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed7 |
| flickr8k-full-local | 7 | t2i_r@5 | 0.8180 | 0.8460 | 0.0280 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed7 |
| flickr8k-full-local | 7 | t2i_r@10 | 0.9020 | 0.9100 | 0.0080 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed7 |
| flickr8k-full-local | 13 | i2t_r@1 | 0.5760 | 0.6050 | 0.0290 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed13 |
| flickr8k-full-local | 13 | i2t_r@5 | 0.8350 | 0.8640 | 0.0290 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed13 |
| flickr8k-full-local | 13 | i2t_r@10 | 0.9140 | 0.9310 | 0.0170 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed13 |
| flickr8k-full-local | 13 | t2i_r@1 | 0.5450 | 0.5840 | 0.0390 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed13 |
| flickr8k-full-local | 13 | t2i_r@5 | 0.8180 | 0.8500 | 0.0320 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed13 |
| flickr8k-full-local | 13 | t2i_r@10 | 0.9020 | 0.9130 | 0.0110 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed13 |
| flickr8k-full-local | 21 | i2t_r@1 | 0.5760 | 0.6090 | 0.0330 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed21 |
| flickr8k-full-local | 21 | i2t_r@5 | 0.8350 | 0.8630 | 0.0280 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed21 |
| flickr8k-full-local | 21 | i2t_r@10 | 0.9140 | 0.9330 | 0.0190 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed21 |
| flickr8k-full-local | 21 | t2i_r@1 | 0.5450 | 0.5820 | 0.0370 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed21 |
| flickr8k-full-local | 21 | t2i_r@5 | 0.8180 | 0.8470 | 0.0290 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed21 |
| flickr8k-full-local | 21 | t2i_r@10 | 0.9020 | 0.9130 | 0.0110 | 245760 | 151523073 | 2 | results\q1_clip_lora_flickr8k_full_seed21 |