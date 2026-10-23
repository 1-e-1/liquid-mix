# Q1 补充实验：统一二阶段重排、统计检验与鲁棒性报告

## 1. 可执行协议

- Observation：冻结 CLIP ViT-B/32 的图像/文本嵌入，形状分别为 `[B, 512]`；第一阶段仅用 cosine similarity 产生 top-K。
- State：ADR 将每个模态投影为 `[B, 128, 4, 4]` 浓度场，显式 Euler 迭代 4 步，`dt=0.2`。
- Action/输出：本任务不是 RL；模型输出候选图文对的标量重排分数。
- Loss：双向 listwise cross-entropy + `0.05` anchor geometry loss + `0.001` residual-scale penalty。
- 候选协议：训练集每个查询 12 个冻结 CLIP hard negatives；测试阶段 top-200，Pokémon 因库大小使用 top-80。
- 训练与选择：固定 seed 13，3 epochs；每个 epoch 仅在 validation 上选择 checkpoint 与 `lambda in [0,2]`（步长 0.05），`lambda=0` 是严格 CLIP 回退。
- 测试隔离：test 只在 checkpoint 与 lambda 固定后读取；逐查询结果用于 paired bootstrap 与 exact McNemar，六项指标内使用 Holm correction。

### 固定划分

| dataset | split | samples | unique_images | manifest_sha256 | protocol |
| --- | --- | --- | --- | --- | --- |
| pokemon | train | 240 | 240 | 0161d3c51a9ffa9260752d8d983517439a89733dce6362d5c86dc9ce42285de4 | fixed project split, 240/80/80 |
| pokemon | val | 80 | 80 | e56c72a9757c7f10ce86948102b396504d541de7779138dde609291063f5996d | fixed project split, 240/80/80 |
| pokemon | test | 80 | 80 | 0cb27c9300d958d599741e3dd840f4e5c6543da8762b301bc4b1f361a5742f08 | fixed project split, 240/80/80 |
| coco1k | train | 5000 | 5000 | 16b33671bafb0ef9f4c73e56df716d321771049c2295e5b8af34bc813fe9e6c1 | fixed controlled subset, 5000/1000/1000 |
| coco1k | val | 1000 | 1000 | a4f4e99270a314eb2594abdf8fb63395af857e96c1bcb5e1757caa1a26394a98 | fixed controlled subset, 5000/1000/1000 |
| coco1k | test | 1000 | 1000 | b6cd110610af90e764a3825cac90e589fc8236bc426c758d3aee01b0bc2be56a | fixed controlled subset, 5000/1000/1000 |
| coco_karpathy | train | 10000 | 10000 | 8a31d33ce85c1e4ce04f7fcec0e1f6e37c36f8a3b3a0ef53b8477ec53ed7ef7e | Karpathy-style fixed split, 10000/5000/5000 |
| coco_karpathy | val | 5000 | 5000 | d47732a15529be7ff5ab9225000e3fe1b4022085da8c6fec3c9d1110c86ca23d | Karpathy-style fixed split, 10000/5000/5000 |
| coco_karpathy | test | 5000 | 5000 | 807b304972f5ffd6c104b69b611ce639cefe9e492553372bcd2405c06fefac9c | Karpathy-style fixed split, 10000/5000/5000 |
| flickr8k | train | 6000 | 6000 | 473a4423d415e55de45d4e3bc556784535e7a281fab42b470bccc7b86409d688 | official fixed split, 6000/1000/1000 |
| flickr8k | val | 1000 | 1000 | 7ace543ede595fa788045c8c5d5faccedcd17fcef9e97a36c668df6342100223 | official fixed split, 6000/1000/1000 |
| flickr8k | test | 1000 | 1000 | 66326c05c0ab83f1db38f52650dd234fc46d3b93482cbabe93646d9f74e34b58 | official fixed split, 6000/1000/1000 |
| flickr30k | train | 10000 | 10000 | d63b94eeafcb6cb4f6b094aca5f5353b889986b73c29c3ee43207ca0f584694c | fixed split, 10000/1000/1000 |
| flickr30k | val | 1000 | 1000 | 3f19d264e23f076e570b00ff898505a8fe73634d981edd2e98222b560514976e | fixed split, 10000/1000/1000 |
| flickr30k | test | 1000 | 1000 | 32da99618bb3ef60fa862687a126e901e2516111292e83dc57fbca0d37b6c8a5 | fixed split, 10000/1000/1000 |

跨 split 图像重叠与 source-ID 重叠均为 0：`True`。

## 2. 二阶段交互基线

新增六个真实 top-K 二阶段头：symmetric MLP、low-rank bilinear、gated multimodal fusion、bidirectional cross-attention、late interaction、Transformer cross-encoder。它们均只重排相同的第一阶段候选，并使用相同训练损失与验证选择。这里是机制级公平实现，不声称逐层复现 SCAN、VSE++ 等原论文。

| model | mean_delta_r1 | mean_delta_r5 | mean_delta_r10 | r1_wins |
| --- | --- | --- | --- | --- |
| ADR FluidMix | 0.0641 | 0.0442 | 0.0263 | 10 |
| Late interaction | 0.0582 | 0.0413 | 0.0239 | 10 |
| Bi-cross-attention | 0.0581 | 0.0417 | 0.0265 | 10 |
| Transformer cross-encoder | 0.0562 | 0.0422 | 0.0222 | 10 |
| Parameter-matched MLP | 0.0509 | 0.0394 | 0.0250 | 10 |
| Low-rank bilinear | 0.0508 | 0.0400 | 0.0207 | 9 |
| Gated fusion | 0.0503 | 0.0397 | 0.0250 | 10 |
| Symmetric MLP | 0.0387 | 0.0382 | 0.0191 | 10 |

## 3. ADR 主结果

| dataset | direction | mix_weight | r@1_clip | r@1 | delta_r@1 | r@5_clip | r@5 | delta_r@5 | r@10_clip | r@10 | delta_r@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pokemon | i2t | 0.1500 | 0.1875 | 0.4375 | 0.2500 | 0.5500 | 0.7250 | 0.1750 | 0.7750 | 0.8000 | 0.0250 |
| pokemon | t2i | 0.1500 | 0.4250 | 0.4750 | 0.0500 | 0.7375 | 0.7250 | -0.0125 | 0.8500 | 0.8500 | 0.0000 |
| coco1k | i2t | 0.1000 | 0.5380 | 0.5630 | 0.0250 | 0.8130 | 0.8180 | 0.0050 | 0.9000 | 0.9190 | 0.0190 |
| coco1k | t2i | 0.1000 | 0.4910 | 0.5430 | 0.0520 | 0.7810 | 0.8170 | 0.0360 | 0.8810 | 0.9050 | 0.0240 |
| coco_karpathy | i2t | 0.1000 | 0.3410 | 0.3646 | 0.0236 | 0.5852 | 0.6254 | 0.0402 | 0.6872 | 0.7276 | 0.0404 |
| coco_karpathy | t2i | 0.1000 | 0.2924 | 0.3414 | 0.0490 | 0.5466 | 0.6068 | 0.0602 | 0.6512 | 0.7144 | 0.0632 |
| flickr8k | i2t | 0.1000 | 0.5760 | 0.6170 | 0.0410 | 0.8350 | 0.8610 | 0.0260 | 0.9140 | 0.9320 | 0.0180 |
| flickr8k | t2i | 0.1000 | 0.5470 | 0.5800 | 0.0330 | 0.8170 | 0.8640 | 0.0470 | 0.9020 | 0.9290 | 0.0270 |
| flickr30k | i2t | 0.1500 | 0.7150 | 0.7600 | 0.0450 | 0.9060 | 0.9380 | 0.0320 | 0.9420 | 0.9690 | 0.0270 |
| flickr30k | t2i | 0.1500 | 0.6830 | 0.7550 | 0.0720 | 0.8950 | 0.9280 | 0.0330 | 0.9440 | 0.9630 | 0.0190 |

ADR 相对 CLIP 的 30 项成对比较中，Holm 校正后显著项为 `25/30`。平均绝对 R@1 提升为 `6.41` 个百分点。

### 与验证集选出的强交互基线和等参数基线比较

| baseline | mean_delta | significant_adr_wins | significant_adr_losses |
| --- | --- | --- | --- |
| Bi-cross-attention | 0.0027 | 1 | 0 |
| Parameter-matched MLP | 0.0064 | 3 | 0 |

## 4. ADR 三算子消融与参数量匹配

| model | r@1 | r@5 | r@10 |
| --- | --- | --- | --- |
| ADR FluidMix | 0.5436 | 0.7908 | 0.8709 |
| ADR w/o advection | 0.5438 | 0.7908 | 0.8710 |
| ADR w/o diffusion | 0.5440 | 0.7908 | 0.8707 |
| ADR w/o reaction | 0.5476 | 0.7912 | 0.8723 |
| Parameter-matched MLP | 0.5305 | 0.7861 | 0.8697 |

完整 ADR 优于等参数 MLP，但三项逐算子消融差异很小；去除 reaction 的平均 R@1 反而略高。当前证据支持 ADR 场结构与参数量无关的收益，但不支持“三个算子都不可替代”的强结论。该负结果应在论文中如实保留。

## 5. 效率

| model | trainable_parameters | estimated_flops_per_pair | peak_inference_vram_mb_topk | peak_training_vram_mb_512_pairs | single_query_topk_latency_median_ms | single_query_topk_latency_p95_ms | training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Low-rank bilinear | 147970 | 295168 | 21.7715 | 23.6025 | 0.3409 | 0.4811 | 5.3706 |
| Bi-cross-attention | 724226 | 3293440 | 25.9209 | 76.6772 | 0.9213 | 1.1742 | 9.9871 |
| ADR FluidMix | 2365186 | 6906112 | 49.2358 | 124.0688 | 2.2326 | 2.4985 | 15.4797 |
| Parameter-matched MLP | 2366127 | 4728958 | 32.9561 | 44.4590 | 0.7518 | 0.9638 | 6.0906 |

FLOPs 为模块钩子覆盖的 Linear、Conv、Attention 与显式 ADR 有限差分运算估计；显存、训练时间和 top-200 延迟为 RTX 5070 Ti 实测。训练时间包含三轮训练与 validation 选模，不包含一次性冻结 CLIP 特征编码。

## 6. 鲁棒性

### 跨域

| model | r@1 | r@5 | r@10 |
| --- | --- | --- | --- |
| ADR FluidMix | 0.6332 | 0.8713 | 0.9293 |
| ADR w/o advection | 0.6332 | 0.8713 | 0.9293 |
| Low-rank bilinear | 0.6332 | 0.8693 | 0.9300 |
| ADR w/o diffusion | 0.6330 | 0.8713 | 0.9297 |
| ADR w/o reaction | 0.6328 | 0.8695 | 0.9305 |
| Gated fusion | 0.6310 | 0.8672 | 0.9305 |
| Late interaction | 0.6263 | 0.8650 | 0.9278 |
| Transformer cross-encoder | 0.6252 | 0.8667 | 0.9278 |
| Parameter-matched MLP | 0.6217 | 0.8627 | 0.9258 |
| Bi-cross-attention | 0.6193 | 0.8675 | 0.9280 |
| Symmetric MLP | 0.6170 | 0.8635 | 0.9263 |
| CLIP | 0.5917 | 0.8412 | 0.9138 |

自然图像目标域上 ADR 与 low-rank bilinear 并列最高；在 Pokémon 极端卡通域偏移上 ADR 低于 CLIP，因此不能宣称任意域都占优。

### 噪声文本

| model | severity | r@1 | r@1_degradation |
| --- | --- | --- | --- |
| ADR FluidMix | 0.1000 | 0.2803 | 0.0727 |
| ADR FluidMix | 0.2000 | 0.2134 | 0.1396 |
| ADR FluidMix | 0.3000 | 0.1565 | 0.1965 |
| Bi-cross-attention | 0.1000 | 0.2758 | 0.0743 |
| Bi-cross-attention | 0.2000 | 0.2093 | 0.1408 |
| Bi-cross-attention | 0.3000 | 0.1563 | 0.1938 |
| CLIP | 0.1000 | 0.2451 | 0.0716 |
| CLIP | 0.2000 | 0.1867 | 0.1300 |
| CLIP | 0.3000 | 0.1389 | 0.1778 |
| Parameter-matched MLP | 0.1000 | 0.2763 | 0.0731 |
| Parameter-matched MLP | 0.2000 | 0.2092 | 0.1402 |
| Parameter-matched MLP | 0.3000 | 0.1544 | 0.1950 |

噪声由 word deletion、adjacent swap 与 internal-character transposition 组成。ADR 在三个强度下的绝对 R@1 均高于 CLIP，但相对 clean 的降幅不是所有模型中最小，合理结论是“噪声下仍保持重排增益”。

### 错配描述

| model | mismatch_auc | balanced_accuracy | rejection_rate_on_mismatch |
| --- | --- | --- | --- |
| ADR w/o advection | 0.9982 | 0.9808 | 0.9864 |
| ADR FluidMix | 0.9982 | 0.9808 | 0.9864 |
| ADR w/o diffusion | 0.9982 | 0.9808 | 0.9864 |
| ADR w/o reaction | 0.9982 | 0.9824 | 0.9848 |
| Gated fusion | 0.9981 | 0.9790 | 0.9824 |
| Transformer cross-encoder | 0.9980 | 0.9814 | 0.9824 |
| Low-rank bilinear | 0.9980 | 0.9796 | 0.9816 |
| Bi-cross-attention | 0.9979 | 0.9796 | 0.9796 |
| Symmetric MLP | 0.9978 | 0.9786 | 0.9772 |
| Late interaction | 0.9977 | 0.9786 | 0.9796 |
| Parameter-matched MLP | 0.9977 | 0.9782 | 0.9800 |
| CLIP | 0.9970 | 0.9762 | 0.9688 |

错配采用固定循环置换（offset 997），validation 选择阈值，test 报告 AUC、balanced accuracy 与 50% 错配时 rejection rate。ADR 的 AUC 与去对流版本并列最高。

## 7. 结论边界

1. 已证实：统一候选与统一选择协议下，ADR 在五数据集的 10 个方向上均提高 R@1，且大多数提升具有成对统计显著性。
2. 已证实：ADR 的主结果高于等参数 MLP，不能仅用参数量增加解释。
3. 部分证实：ADR 在自然图像跨域、噪声文本和可控错配下保持强表现；极端 Pokémon 域偏移失败。
4. 未证实：reaction 算子在当前超参数下不可替代。消融结果提示下一轮应重新研究 reaction 的稳定化形式或系数约束。

## 8. 最终文件

- `aggregate_metrics.csv`：五数据集所有测试结果。
- `statistics_adr_vs_selected_baselines.csv`：ADR 对 cross-attention 与等参数 MLP 的 paired bootstrap/McNemar/Holm。
- `statistics_adr_vs_ablations.csv`：ADR 三算子逐项比较。
- `efficiency.csv`：参数、FLOPs、显存、时间与延迟。
- `robustness/`：跨域、噪声文本、错配描述结果与协议。
- `protocol/`：固定划分 hash、重叠审计与候选缓存。
- `figures/`：最终论文级结果图。

## 参考机制

- Faghri et al., VSE++: Improving Visual-Semantic Embeddings with Hard Negatives, 2018: https://arxiv.org/abs/1707.05612
- Lee et al., Stacked Cross Attention for Image-Text Matching, ECCV 2018: https://openaccess.thecvf.com/content_ECCV_2018/html/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.html
- Zhang et al., Negative-Aware Attention Framework for Image-Text Matching, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Negative-Aware_Attention_Framework_for_Image-Text_Matching_CVPR_2022_paper.html