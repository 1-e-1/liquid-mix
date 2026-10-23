# FluidMix: A Liquid Advection-Diffusion-Reaction Fusion Module for Vision-Language Matching

## Abstract

Vision-language models (VLMs) commonly represent each modality as a static embedding and perform fusion through concatenation, attention, or contrastive similarity. This paper studies a complementary hypothesis: a modality representation can be modeled as a concentration field, and multimodal fusion can be formulated as a continuous-time mixing process governed by advection, diffusion, and reaction dynamics. We introduce FluidMix, a lightweight fusion module that converts image and text representations into concentration fields and evolves their interaction through an explicit discretization of an advection-diffusion-reaction equation. We evaluate three forms of the model: an ordinary FluidMix head trained on frozen CLIP embeddings, a residual FluidMix scorer anchored to CLIP cosine similarity, and a calibrated residual FluidMix scorer whose blending coefficient is selected on validation data. We also implement two CLIP-free variants: a fully project-local Pure FluidMix VLM and a ResNet50+MiniLM+FluidMix baseline. Experiments on Pokemon-BLIP, Flickr8k, and Flickr30k show that residual anchoring is critical. Ordinary FluidMix and Pure FluidMix do not replace CLIP under the current small-data regime. In contrast, calibrated residual FluidMix improves the point estimates of test accuracy and image-to-text R@1 over CLIP zero-shot on all three evaluated datasets. On Pokemon-BLIP, paired bootstrap analysis gives strong evidence for retrieval improvement, with R@1 increasing from 0.1875 to 0.4000 and R@5 from 0.5625 to 0.7250. On Flickr8k and Flickr30k subsets, the improvements are smaller and most confidence intervals cross zero, indicating trends rather than statistically decisive gains. These results position FluidMix not as a complete replacement for large-scale contrastive pretraining, but as a physically inspired residual fusion mechanism that can improve or calibrate frozen VLM representations.

**Keywords:** vision-language model, multimodal fusion, liquid neural network, advection-diffusion-reaction equation, CLIP, image-text retrieval.

## 1. Introduction

Large-scale VLMs such as CLIP learn transferable image-text representations by contrastive pretraining on hundreds of millions of image-caption pairs [@radford2021clip]. Their success has shaped a dominant design pattern: encode images and text into vectors, align them with a contrastive objective, and reuse cosine similarity for zero-shot matching or retrieval. This paradigm is powerful, but it makes the fusion operation largely static at inference time. Once embeddings are computed, the cross-modal interaction is often reduced to vector similarity, concatenation, or attention over token sequences.

This paper investigates a different modeling assumption. Instead of treating each modality as a static vector, we interpret each modality as a concentration field. Multimodal fusion is then formulated as a liquid mixing process: image and text concentrations are combined, transported, smoothed, and transformed by an advection-diffusion-reaction dynamical system. This view is motivated by two lines of work. First, continuous-depth and liquid neural models have shown that differential equations can provide expressive temporal dynamics and adaptive state evolution [@chen2018neuralode; @hasani2021liquid]. Second, reaction-diffusion and transport equations provide a compact mathematical language for local propagation, smoothing, and nonlinear interaction in spatial fields [@turing1952morphogenesis].

We propose FluidMix, a compact module that lifts image and text representations into modality-specific concentration fields and evolves the fused field using a finite-step explicit solver. FluidMix can be attached to frozen CLIP embeddings, used as a residual correction to CLIP cosine similarity, or trained in a CLIP-free VLM architecture. This paper focuses on a realistic question for research engineering: does the fluid-inspired fusion mechanism improve multimodal matching beyond a strong CLIP zero-shot baseline, and under what conditions?

The answer from our experiments is nuanced. A direct ordinary FluidMix head is not sufficient; it substantially underperforms CLIP on all evaluated datasets. A Pure FluidMix model trained without CLIP also fails to match CLIP under the present small-data regime. However, a residual form initialized to exactly preserve CLIP ranking and trained only as a correction term is substantially more stable. When combined with validation-set calibration, the residual model improves point estimates of accuracy and R@1 over CLIP zero-shot on Pokemon-BLIP, Flickr8k, and Flickr30k. The strongest evidence appears in Pokemon-BLIP image-to-text retrieval, where paired bootstrap confidence intervals do not cross zero. On Flickr8k and Flickr30k subsets, the gains are small and should be interpreted as preliminary.

The contributions are:

1. We formulate multimodal image-text fusion as a learnable advection-diffusion-reaction process over modality concentration fields.
2. We implement ordinary, residual, calibrated residual, and fully CLIP-free FluidMix variants in a reproducible experimental codebase.
3. We compare FluidMix against CLIP zero-shot, classical fusion adapters, a liquid-state baseline, and CLIP-free backbones on three datasets.
4. We provide paired bootstrap analysis and explicitly distinguish statistically supported conclusions from point-estimate trends.

## 2. Background

### 2.1 Vision-language contrastive pretraining

CLIP trains image and text encoders through a large-scale contrastive objective, yielding embeddings whose cosine similarity can be used for zero-shot classification and retrieval [@radford2021clip]. Its common vision backbones include ResNet-style encoders and Vision Transformers [@he2016resnet; @dosovitskiy2021vit]. Similar large-scale contrastive or bootstrapped approaches include ALIGN [@jia2021align], ALBEF [@li2021albef], and BLIP [@li2022blip]. These models demonstrate the importance of large-scale paired data and contrastive alignment. In the present work, CLIP is both a strong baseline and, in residual variants, a frozen representation backbone.

### 2.2 Continuous-time neural dynamics

Neural Ordinary Differential Equations model hidden-state evolution as an initial value problem [@chen2018neuralode]. Liquid Time-Constant Networks extend this idea through input-dependent time constants and adaptive dynamics, showing that continuous-time parameterizations can provide compact and expressive models for sequential signals [@hasani2021liquid]. FluidMix borrows this continuous-time perspective but applies it to multimodal fusion fields rather than temporal sensor streams.

### 2.3 Advection-diffusion-reaction systems

Advection-diffusion-reaction equations model transport, smoothing, and local nonlinear transformation:

```math
\frac{\partial C}{\partial t}
= -v \cdot \nabla C + D \nabla^2 C + R(C).
```

Here, `C` is a concentration field, `v` is a velocity field, `D` is a diffusion coefficient, and `R` is a reaction term. Reaction-diffusion systems have a long history in mathematical biology and pattern formation [@turing1952morphogenesis]. We use this equation as an inductive bias for multimodal interaction: advection models directed cross-modal transport, diffusion models smoothing and local agreement, and reaction models nonlinear feature transformation.

## 3. Related Work

### 3.1 Vision-language models

Earlier VLMs such as ViLBERT and VisualBERT used transformer-based cross-modal interaction for vision-language reasoning [@lu2019vilbert; @li2019visualbert]. CLIP and ALIGN shifted the focus toward large-scale contrastive representation learning [@radford2021clip; @jia2021align]. ALBEF and BLIP further improved alignment and bootstrapping for understanding and generation [@li2021albef; @li2022blip]. These models typically rely on attention, contrastive losses, or encoder-decoder objectives rather than explicit physical field dynamics.

### 3.2 Multimodal fusion mechanisms

Classical multimodal fusion includes concatenation, gating, bilinear pooling, tensor fusion, and low-rank fusion [@zadeh2017tfn; @liu2018lowrank; @fukui2016mcb]. Cross-attention remains a standard mechanism for high-capacity interaction. FluidMix differs by representing modalities as spatial concentration fields and applying local differential operators. It is not intended to replace attention universally; instead, it offers a compact physically inspired adapter that can be combined with existing encoders.

### 3.3 Differential-equation-inspired neural modules

Differential-equation-based models provide continuous-depth networks, adaptive dynamics, and physically inspired constraints [@chen2018neuralode; @hasani2021liquid; @rackauckas2020ude]. Physics-informed neural networks and data-driven PDE discovery further demonstrate the value of combining learnable models with differential-equation priors [@raissi2019pinn; @rudy2017pde]. FluidMix follows this broad philosophy but targets image-text fusion rather than physical simulation.

## 4. Model Design

### 4.1 Notation and task

Each sample consists of an image `I_i` and caption `T_i`. The supervised matching task receives a pair `(I_i,T_j)` and predicts a compatibility score `s_{ij}`. In retrieval, the model constructs a score matrix `S in R^{N x N}` for `N` images and `N` captions, where diagonal entries are positive matches.

This is not a reinforcement learning or robotic control setting. Observation is the image-caption pair. The model state is the internal concentration field. There is no action, reward, or control frequency.

### 4.2 Frozen CLIP feature interface

For CLIP-based variants, frozen encoders produce:

```math
z_I = E_I(I) \in R^{512}, \quad z_T = E_T(T) \in R^{512}.
```

The zero-shot baseline is:

```math
s_{\text{CLIP}}(I,T) =
\text{cos}(z_I,z_T).
```

FluidMix maps embeddings into both global vectors and spatial fields:

```math
h_I = P_I z_I,\quad h_T = P_T z_T,
```

```math
C_I = \tanh(W_I z_I) \in R^{d \times G \times G},\quad
C_T = \tanh(W_T z_T) \in R^{d \times G \times G}.
```

In the main experiments, `d=128` and `G=4`, so the concentration field shape is `[B,128,4,4]`.

### 4.3 Ordinary FluidMix

Ordinary FluidMix predicts:

```math
s_{\text{FM}}(I,T) = q_\theta(F_\theta(h_I,h_T,C_I,C_T)),
```

where `F_theta` is the FluidMix dynamical module and `q_theta` is a classifier head. This variant uses frozen CLIP embeddings but does not preserve CLIP cosine similarity.

### 4.4 Residual FluidMix

Residual FluidMix anchors the model to CLIP:

```math
s_{\text{res}}(I,T)
= s_{\text{CLIP}}(I,T) + \alpha q_\theta(F_\theta(h_I,h_T,C_I,C_T)).
```

The scalar `alpha` is initialized to zero. Therefore, before training:

```math
s_{\text{res}}(I,T) = s_{\text{CLIP}}(I,T).
```

This initialization is crucial. It prevents the adapter from destroying CLIP's pretrained ranking at the beginning of training and allows FluidMix to learn only a residual correction.

### 4.5 Calibrated residual FluidMix

For calibrated residual FluidMix, validation data select a blending coefficient `lambda`:

```math
s_{\text{cal}}(I,T;\lambda)
= (1-\lambda)s_{\text{CLIP}}(I,T)
 + \lambda s_{\text{res}}(I,T).
```

The coefficient is selected using only validation metrics. Test labels are not used for calibration.

### 4.6 Pure FluidMix without CLIP

The Pure FluidMix variant removes CLIP completely. Images are encoded by a project-local CNN patch encoder:

```math
I \mapsto (h_I, C_I),
```

and captions are encoded by a project-local tokenizer and transformer text encoder:

```math
T \mapsto (h_T, C_T).
```

The score combines a learned retrieval logit and a FluidMix logit. This variant tests whether the fluid-inspired structure alone can replace CLIP under the current data regime.

### 4.7 CLIP-free pretrained backbone baseline

We also evaluate a non-CLIP pretrained baseline using ImageNet ResNet50 for images and MiniLM/Sentence-BERT-style sentence embeddings for text [@he2016resnet; @reimers2019sentencebert]. This controls for the difference between "no CLIP" and "no pretraining".

## 5. Mathematical Derivation

### 5.1 Initial mixture

Let `C_I` and `C_T` be modality concentration fields. A learned gate defines the initial mixture:

```math
g = \sigma(W_g[h_I;h_T]), \quad
C^0 = g \odot C_I + (1-g) \odot C_T.
```

The gate is channel-wise and broadcast over spatial dimensions.

### 5.2 Continuous dynamics

We model fusion as:

```math
\frac{\partial C}{\partial t}
= -v_x \frac{\partial C}{\partial x}
   -v_y \frac{\partial C}{\partial y}
   + D \nabla^2 C
   + \rho \tanh(W_r * C),
```

where `v_x` and `v_y` are learned channel-wise velocities from `[h_I;h_T]`, `D` is the diffusion coefficient, `rho` is the reaction strength, and `W_r` is a `1 x 1` convolution.

The interpretation is:

- Advection transports modality evidence across the field.
- Diffusion smooths local disagreement.
- Reaction applies nonlinear cross-channel transformation.

### 5.3 Discretization

For grid index `(u,v)` and channel `c`, the central finite differences are:

```math
\delta_x C_{c,u,v}
= \frac{C_{c,u,v+1} - C_{c,u,v-1}}{2},
```

```math
\delta_y C_{c,u,v}
= \frac{C_{c,u+1,v} - C_{c,u-1,v}}{2}.
```

The five-point Laplacian is:

```math
\Delta C_{c,u,v}
= C_{c,u+1,v} + C_{c,u-1,v}
 + C_{c,u,v+1} + C_{c,u,v-1}
 - 4C_{c,u,v}.
```

The implementation uses explicit Euler integration:

```math
C^{k+1} = C^k + \Delta t
\left[
-v_x \delta_x C^k
-v_y \delta_y C^k
+D\Delta C^k
+\rho \tanh(W_r * C^k)
\right].
```

The default solver uses `K=4` steps and `Delta t=0.2`.

### 5.4 Fused representation

After `K` steps, the field is pooled into a vector:

```math
f = \text{LayerNorm}\left(\frac{1}{G^2}\sum_{u,v} C^K_{:,u,v}\right).
```

The score head maps `f` to a scalar matching logit.

### 5.5 Loss functions

Binary image-caption matching uses:

```math
L_{\text{BCE}}
= -y\log \sigma(s) - (1-y)\log(1-\sigma(s)).
```

For a batch score matrix `S`, symmetric contrastive retrieval uses:

```math
L_{\text{ITC}}
= \frac{1}{2}\text{CE}(S/\tau, \text{diag})
 + \frac{1}{2}\text{CE}(S^T/\tau, \text{diag}).
```

The total loss is:

```math
L = L_{\text{BCE}} + \beta L_{\text{ITC}}.
```

## 6. Experiments

### 6.1 Datasets

We evaluate on three image-caption datasets:

| Dataset | Train | Val | Test | Notes |
| --- | ---: | ---: | ---: | --- |
| Pokemon-BLIP | 240 | 80 | 80 | `reach-vb/pokemon-blip-captions` |
| Flickr8k | 1000 | 200 | 200 | local ZIP subset using official split |
| Flickr30k | 1000 | 200 | 200 | local ZIP subset with deterministic split |

Flickr8k and Flickr30k are standard image-caption datasets [@hodosh2013flickr8k; @young2014flickr30k]. MS COCO and COCO Captions are also widely used for full-scale image-caption evaluation [@lin2014coco; @chen2015cococaptions], but they are left for future full-benchmark expansion. The present experiments use subsets to make multi-model training feasible on the available workstation. Therefore, the Flickr results should be interpreted as controlled local evidence, not as a full benchmark claim.

### 6.2 Compared models

We compare:

- CLIP zero-shot cosine.
- Ordinary FluidMix on frozen CLIP embeddings.
- Residual FluidMix.
- Calibrated residual FluidMix.
- Pure FluidMix without CLIP.
- ResNet50+MiniLM+FluidMix without CLIP.
- Adapter ablations: concat, gated fusion, cross-attention, and liquid fusion.

### 6.3 Hyperparameters

Main FluidMix parameters:

| Parameter | Value |
| --- | ---: |
| hidden dimension | 128 |
| field grid | 4 x 4 |
| integration steps | 4 |
| dt | 0.2 |
| diffusion | 0.03 |
| reaction | 0.5 |
| CLIP model | `openai/clip-vit-base-patch32` |
| main seed | 13 |
| split seed | 42 |

The CLIP adapter uses hard negatives from CLIP cosine top-ranked non-matching captions. Evaluation uses validation-selected binary thresholds and test-set retrieval matrices.

### 6.4 Metrics

Binary matching metrics:

- Accuracy.
- ROC-AUC.

Retrieval metrics:

- Image-to-text R@1.
- Image-to-text R@5.

For calibrated residual FluidMix versus CLIP zero-shot, paired bootstrap estimates confidence intervals for accuracy, R@1, and R@5.

### 6.5 Main results

| Dataset | Model | Uses CLIP | Accuracy | AUC | R@1 | R@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Pokemon-BLIP | CLIP zero-shot | yes | 0.8000 | 0.8683 | 0.1875 | 0.5625 |
| Pokemon-BLIP | ordinary FluidMix | frozen embeddings | 0.4813 | 0.5022 | 0.0250 | 0.0375 |
| Pokemon-BLIP | residual FluidMix | frozen CLIP + residual | 0.8187 | 0.8775 | 0.3750 | 0.7125 |
| Pokemon-BLIP | calibrated residual FluidMix | frozen CLIP + calibration | 0.8187 | 0.8752 | 0.4000 | 0.7250 |
| Pokemon-BLIP | Pure FluidMix | no | 0.5250 | 0.5830 | 0.0250 | 0.1125 |
| Pokemon-BLIP | ResNet50+MiniLM+FluidMix | no | 0.6813 | 0.7516 | 0.0750 | 0.2500 |
| Flickr8k | CLIP zero-shot | yes | 0.9725 | 0.9899 | 0.7600 | 0.9650 |
| Flickr8k | ordinary FluidMix | frozen embeddings | 0.6325 | 0.6378 | 0.0050 | 0.0500 |
| Flickr8k | residual FluidMix | frozen CLIP + residual | 0.9325 | 0.9823 | 0.5950 | 0.9150 |
| Flickr8k | calibrated residual FluidMix | frozen CLIP + calibration | 0.9750 | 0.9919 | 0.7800 | 0.9600 |
| Flickr8k | Pure FluidMix | no | 0.5950 | 0.6155 | 0.0000 | 0.0350 |
| Flickr8k | ResNet50+MiniLM+FluidMix | no | 0.8650 | 0.9355 | 0.1500 | 0.4300 |
| Flickr30k | CLIP zero-shot | yes | 0.9750 | 0.9977 | 0.8900 | 0.9900 |
| Flickr30k | ordinary FluidMix | frozen embeddings | 0.5900 | 0.6388 | 0.0100 | 0.0600 |
| Flickr30k | residual FluidMix | frozen CLIP + residual | 0.9625 | 0.9877 | 0.7600 | 0.9500 |
| Flickr30k | calibrated residual FluidMix | frozen CLIP + calibration | 0.9800 | 0.9968 | 0.8950 | 0.9950 |
| Flickr30k | Pure FluidMix | no | 0.5100 | 0.5518 | 0.0050 | 0.0200 |
| Flickr30k | ResNet50+MiniLM+FluidMix | no | 0.8250 | 0.8970 | 0.2050 | 0.4350 |

The results show three clear patterns. First, ordinary FluidMix is weak because it discards the CLIP cosine anchor. Second, Pure FluidMix is not competitive with CLIP under small-data training. Third, calibrated residual FluidMix is the strongest variant: it improves accuracy and R@1 point estimates over CLIP zero-shot on all three datasets.

### 6.6 Difference from CLIP zero-shot

| Dataset | Model | Delta Acc. | Delta AUC | Delta R@1 | Delta R@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Pokemon-BLIP | calibrated residual FluidMix | +0.0187 | +0.0069 | +0.2125 | +0.1625 |
| Flickr8k | calibrated residual FluidMix | +0.0025 | +0.0020 | +0.0200 | -0.0050 |
| Flickr30k | calibrated residual FluidMix | +0.0050 | -0.0008 | +0.0050 | +0.0050 |

The Pokemon-BLIP gain is large in retrieval. Flickr gains are small and should be treated as preliminary.

### 6.7 Paired bootstrap analysis

| Dataset | Metric | CLIP | Calibrated | Delta | 95% CI | P(Delta > 0) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Pokemon-BLIP | Accuracy | 0.8000 | 0.8187 | +0.0187 | [-0.0250, +0.0687] | 0.7411 |
| Pokemon-BLIP | R@1 | 0.1875 | 0.4000 | +0.2125 | [+0.1125, +0.3125] | 1.0000 |
| Pokemon-BLIP | R@5 | 0.5625 | 0.7250 | +0.1625 | [+0.0750, +0.2625] | 0.9994 |
| Flickr8k | Accuracy | 0.9725 | 0.9750 | +0.0025 | [-0.0075, +0.0150] | 0.5833 |
| Flickr8k | R@1 | 0.7600 | 0.7800 | +0.0200 | [-0.0200, +0.0600] | 0.8035 |
| Flickr8k | R@5 | 0.9650 | 0.9600 | -0.0050 | [-0.0250, +0.0100] | 0.1865 |
| Flickr30k | Accuracy | 0.9750 | 0.9800 | +0.0050 | [-0.0075, +0.0175] | 0.7280 |
| Flickr30k | R@1 | 0.8900 | 0.8950 | +0.0050 | [-0.0250, +0.0350] | 0.5678 |
| Flickr30k | R@5 | 0.9900 | 0.9950 | +0.0050 | [+0.0000, +0.0150] | 0.6279 |

Only the Pokemon-BLIP retrieval improvements provide strong bootstrap evidence. Accuracy improvement on Pokemon-BLIP and most Flickr improvements have confidence intervals that cross zero.

### 6.8 Ablation findings

The adapter ablation shows that residual anchoring is more important than the specific nonlinear head. Concat, gated, cross-attention, liquid fusion, and ordinary FluidMix all degrade sharply when they replace CLIP cosine directly. Therefore, the main engineering lesson is not merely that "fluid dynamics helps"; it is that physically inspired dynamics must be integrated in a residual or calibrated manner when a powerful pretrained VLM already provides a strong ranking prior.

## 7. Discussion

The experiments support a bounded claim. FluidMix is useful as a residual fusion module on top of frozen CLIP, especially for small-domain adaptation where the frozen similarity can be corrected without destroying pretrained alignment. It is not currently a complete replacement for CLIP. The Pure FluidMix result is a negative but important finding: a physically inspired fusion equation does not by itself reproduce the effect of large-scale image-text pretraining.

This distinction matters for future work. If the goal is CLIP replacement, the correct next step is not only adapter tuning but pretraining a CLIP-free FluidMix model at scale. If the goal is practical VLM adaptation, residual FluidMix is already promising because it is lightweight, preserves the baseline at initialization, and can be calibrated with validation data.

## 8. Limitations

1. Flickr8k and Flickr30k experiments use subsets rather than the full benchmark splits.
2. Most experiments use a single seed except the Pokemon residual/calibrated seed sweep.
3. The CLIP-free Pure FluidMix model is trained from small data and is not a fair replacement for CLIP-scale pretraining.
4. The current retrieval evaluation reports image-to-text R@1/R@5, not all standard full-benchmark retrieval metrics such as R@10 and text-to-image retrieval.
5. The discretization uses a small `4 x 4` concentration grid. Larger grids and adaptive solvers may change the trade-off between expressivity and stability.

## 9. Conclusion

We presented FluidMix, a liquid advection-diffusion-reaction fusion module for image-text matching. The model treats image and text features as concentration fields and fuses them through learnable transport, diffusion, and reaction dynamics. Across Pokemon-BLIP, Flickr8k, and Flickr30k, the results show that direct replacement of CLIP cosine is unstable, while residual anchoring and validation calibration produce the strongest behavior. Calibrated residual FluidMix improves accuracy and R@1 point estimates over CLIP zero-shot on all evaluated datasets, with strong paired-bootstrap evidence for retrieval improvement on Pokemon-BLIP. The CLIP-free variants show that FluidMix is not yet a complete replacement for large-scale contrastive VLM pretraining. The most defensible conclusion is that FluidMix is a promising physically inspired residual fusion mechanism for frozen VLM adaptation, and a candidate architecture for future large-scale CLIP-free pretraining.

## Data and Code Availability

The implementation, processed local datasets, trained adapters, result tables, and visualizations are organized under `T:/work_space/Liquid_mix`. The main reproducible summary is generated by:

```powershell
py scripts\summarize_multidataset_results.py --root T:\work_space\Liquid_mix --output T:\work_space\Liquid_mix\results\multidataset_summary
```

## References

References are provided in `references.bib`.
