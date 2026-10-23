# Liquid_mix 实验协议

## 研究问题

本工程验证“液态流体混合”是否能提升图文多模态融合能力。核心假设是：图像和文本可以表示为连续浓度场 `C_img` 与 `C_txt`，融合过程由对流、扩散、反应动力学完成。

## 任务定义

- Observation：RGB image `I` 与 caption text `T`。
- State：模型内部状态为浓度场 `C_t`，主实验默认 shape `[B,128,4,4]`。
- Action：本实验不是 RL/VLA 控制任务，没有机器人 action；模型输出图文匹配分数。
- Reward：不使用 RL reward；训练信号来自监督式 matching label 与 retrieval 对角目标。
- Control frequency：不适用。

## 数据集

| dataset | train | val | test | 说明 |
| --- | ---: | ---: | ---: | --- |
| Pokemon-BLIP | 240 | 80 | 80 | `reach-vb/pokemon-blip-captions` |
| Flickr8k | 1000 | 200 | 200 | 本地 ZIP 子集，官方 split |
| Flickr30k | 1000 | 200 | 200 | 本地 ZIP 子集，deterministic split |

## 模型族

- `clip_zero_shot_cosine`：冻结 CLIP image/text encoder，直接 cosine。
- `clip_frozen_fluidmix`：普通 FluidMix adapter，使用冻结 CLIP embedding，但不保留 CLIP cosine anchor。
- `clip_frozen_residual_fluidmix`：`score = CLIP cosine + scale * FluidMix residual`。
- `clip_frozen_calibrated_residual_fluidmix`：validation set 选择混合系数的 residual FluidMix。
- `pure_fluidmix_vlm`：完全无 CLIP，从项目内 CNN + tokenizer/Transformer + FluidMix 训练。
- `clipfree_resnet50_minilm_fluidmix`：无 CLIP，但使用 ImageNet ResNet50 与 MiniLM 单模态预训练 backbone。

## FluidMix 动力学

```text
C_0 = gate * C_img + (1 - gate) * C_txt

C_{t+1} = C_t + dt * (
    - v_x * grad_x(C_t)
    - v_y * grad_y(C_t)
    + diffusion * laplacian(C_t)
    + reaction * tanh(conv1x1(C_t))
)
```

默认参数：

- `steps = 4`
- `dt = 0.2`
- `diffusion = 0.03`
- `reaction = 0.5`
- `field_grid = 4`
- `hidden_dim = 128`

## Loss

Binary matching：

```text
L_bce = BCEWithLogits(score(I_i, T_j), y_ij)
```

Symmetric contrastive retrieval：

```text
L_contrastive = 0.5 * CE(score_matrix / tau, labels)
              + 0.5 * CE(score_matrix.T / tau, labels)
```

总损失：

```text
L = L_bce + contrastive_weight * L_contrastive
```

## Evaluation

- Binary matching：validation set 选择 threshold，test set 固定 threshold 计算 accuracy/AUC。
- Image-to-text retrieval：对 test set 构造 `[N,N]` score matrix，对角线为正匹配，计算 R@1/R@5。
- Statistical check：calibrated residual 与 CLIP zero-shot 使用 paired bootstrap；当前完整 bootstrap 表在 `results/multidataset_summary/paired_bootstrap.csv`。

## 结论边界

当前可支持的强结论：

- Pokemon-BLIP 上，calibrated residual FluidMix 的 retrieval 明显超过 CLIP zero-shot。
- Flickr8k/Flickr30k 子集上，calibrated residual 的 accuracy/R@1 点估计超过 CLIP，但统计证据不足。

当前不能支持的结论：

- 不能说 Pure FluidMix 已经完全替代 CLIP。
- 不能说 ordinary FluidMix 全面超过 CLIP。
- 不能把 Flickr 子集结果写成完整 benchmark 结论。
