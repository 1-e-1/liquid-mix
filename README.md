# Liquid_mix

> Public export, 2026-10-24: source, configurations, tests, research text and
> selected aggregate result tables are provided. Original datasets, feature caches,
> model weights and private working documents are not distributed. Historical
> metrics below are retained from the original project and were not rerun for this
> release. See [publication notes](PUBLICATION_NOTES.md).


本项目研究 FluidMix 作为冻结 VLM 第一阶段检索器之后的轻量二阶段重排器。当前主线不再尝试用小规模训练替代 CLIP，而是保留 CLIP cosine anchor，在其 top-K 候选内使用 Advection-Diffusion-Reaction（ADR）场动力学进行残差重排。

## 当前结论

- 五个固定数据集、两个检索方向上，ADR 的 R@1 均高于冻结 CLIP，平均绝对提升 6.41 个百分点。
- 30 项 Recall 比较中，25 项经 paired bootstrap、exact McNemar 和 Holm correction 后显著。
- ADR 的平均 R@1 高于六个统一协议的二阶段交互基线及等参数 MLP。
- 三算子消融显示去除 reaction 后平均 R@1 略高，因此当前结果不能证明三个算子都不可替代。
- 自然图像跨域、噪声文本和可控错配下 ADR 保持较强表现；极端 Pokémon 域偏移是明确边界。

完整结论见：

`results/q1_extended/Q1_extended_experiment_report_CN.md`

## 统一实验

五个数据集：

- Pokémon-BLIP：240/80/80
- COCO-1K：5000/1000/1000
- COCO Karpathy：10000/5000/5000
- Flickr8k：6000/1000/1000
- Flickr30k：10000/1000/1000

所有 split 均记录 manifest SHA-256，并确认 train/validation/test 的图像与 source ID 零重叠。

二阶段模型：

- Symmetric MLP
- Low-rank bilinear interaction
- Gated multimodal fusion
- Bidirectional cross-attention
- Late interaction
- Transformer cross-encoder
- ADR FluidMix
- ADR w/o advection、w/o diffusion、w/o reaction
- Parameter-matched MLP

统一协议：

- 冻结 CLIP ViT-B/32 产生第一阶段 top-200；Pokémon 使用 top-80。
- 训练 hard negatives 只从 train split 构造。
- checkpoint 与混合权重仅由 validation 选择，`lambda=0` 可严格回退到 CLIP。
- test 只在模型和 lambda 固定后读取。
- 固定 seed 13；逐查询统计使用 10,000 次 bootstrap。

## 复现入口

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行五数据集完整实验：

```powershell
python scripts\run_extended_q1_experiments.py --stage all --device cuda --resume
```

运行跨域、噪声文本和错配描述实验：

```powershell
python scripts\run_q1_robustness.py --source coco_karpathy --device cuda
```

测量参数量、FLOPs、显存、训练时间和 top-200 延迟：

```powershell
python scripts\measure_q1_efficiency.py --dataset coco_karpathy --device cuda
```

重建最终中文报告、统计表和五张结果图：

```powershell
python scripts\build_q1_extended_report.py --bootstrap 5000
```

运行测试：

```powershell
python -m pytest -q
```

## 关键结果

- `results/q1_extended/aggregate_metrics.csv`
- `results/q1_extended/statistics_adr_vs_selected_baselines.csv`
- `results/q1_extended/statistics_adr_vs_ablations.csv`
- `results/q1_extended/efficiency.csv`
- `results/q1_extended/robustness/`
- `results/q1_extended/protocol/`
- `results/q1_extended/figures/`

每个 dataset/model 运行目录均包含 checkpoint、曲线、测试指标、逐查询指示量和相对 CLIP 的显著性检验。

## 保留脚本

- `run_extended_q1_experiments.py`：统一二阶段基线、ADR 与消融主实验。
- `run_q1_robustness.py`：跨域、噪声文本和错配描述压力测试。
- `measure_q1_efficiency.py`：效率与资源测量。
- `build_q1_extended_report.py`：最终结果汇总与绘图。
- `run_clip_vlm_experiment.py`：冻结 CLIP 特征编码与早期适配实验。
- `run_clip_lora_baseline.py`：CLIP-LoRA。
- `run_foundation_vlm_benchmark.py`：OpenCLIP、SigLIP、BLIP 基准。
- `run_pure_fluid_experiment.py`：Pure FluidMix 边界实验。

旧的文档生成、绘图、candidate residual、calibration 和 complete-ablation 中间脚本已经移除；历史论文、Word 文档和正式结果仍保留。
