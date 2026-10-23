# FluidMix: Residual Advection-Diffusion-Reaction Field Dynamics for Vision-Language Retrieval

## Abstract

Vision-language models (VLMs) usually reduce an image and a text input to two static embedding vectors and compare them with a dot product, a learned matching head, or a cross-modal transformer. This paper studies a complementary hypothesis: each modality can be interpreted as a concentration field, and multimodal fusion can be implemented as a finite-step advection-diffusion-reaction mixing process. We propose FluidMix, a lightweight field-dynamic adapter that lifts frozen image and text representations into modality concentration fields, evolves a gated mixed field with explicit Euler updates, and produces a residual correction to CLIP-style image-text similarity. The residual form is initialized to exactly preserve CLIP rankings, and a validation-calibrated variant selects the final residual strength without using test labels. We give code-aligned mathematical derivations for the finite-step dynamics, including Lipschitz perturbation bounds, finite-step boundedness, residual ranking preservation, validation-grid calibration bounds, and the top-k reranking ceiling. We evaluate FluidMix against CLIP zero-shot, OpenCLIP, SigLIP, BLIP-ITM, CLIP-LoRA, ordinary FluidMix, residual FluidMix, calibrated residual FluidMix, Pure FluidMix without CLIP, and non-CLIP pretrained-backbone FluidMix baselines. Experiments cover Pokemon-BLIP, Flickr8k, Flickr30k, COCO-1K, local COCO-5K, COCO Karpathy 5k, Flickr8k-full, and Flickr30k-large protocols. The strongest small-domain FluidMix evidence appears on Pokemon-BLIP, where calibrated residual FluidMix improves image-to-text R@1 from 0.1875 to 0.4000 with paired-bootstrap confidence intervals above zero. The original pairwise-trained residual recipe fails on standard COCO/Flickr benchmarks, but this failure motivates a hard-candidate retrieval residual objective that trains on CLIP top-ranked nonmatches with a CLIP score-geometry regularizer. With validation-selected calibration and CLIP top-200 reranking, this variant improves COCO-5K over CLIP across three seeds in both directions: mean image-to-text R@1/R@5/R@10 deltas are +0.0230/+0.0316/+0.0310, and text-to-image deltas are +0.0383/+0.0599/+0.0578, with positive paired-bootstrap lower bounds for every COCO-5K seed and metric. Under multi-reference five-caption evaluation, the same three seeds improve all six recall metrics on both the local COCO-5K list and the standard COCO Karpathy 5k test split. On COCO Karpathy 5k, mean image-to-text deltas are +0.0209/+0.0240/+0.0166 and text-to-image deltas are +0.0490/+0.0542/+0.0521, with positive worst-case bootstrap lower bounds across seeds. The same hard-candidate protocol gives positive mean deltas on Flickr8k-full and Flickr30k-large. A matched three-seed CLIP-LoRA baseline also gives consistent gains over zero-shot CLIP on COCO-1K, Flickr8k-full, and Flickr30k-large, with all 54 per-seed recall deltas positive; SigLIP remains the strongest embedding baseline among the downloaded foundation models. The main contribution is therefore not a universal replacement for CLIP, but a mathematically specified, reproducible, physically inspired residual adapter study that identifies when field dynamics help, when they fail, and which objective changes make them useful for calibrated standard retrieval reranking.

**Keywords:** vision-language model, multimodal fusion, CLIP, liquid neural network, advection-diffusion-reaction dynamics, image-text retrieval, residual adapter, parameter-efficient fine-tuning.

## 1. Introduction

Large-scale VLMs such as CLIP learn transferable image and text encoders through contrastive pretraining over web-scale image-caption data [@radford2021clip]. After pretraining, many downstream retrieval systems use a simple normalized dot product between image and text embeddings. This design is strong, simple, and difficult to beat without either comparable pretraining scale or a careful downstream adaptation strategy. It also leaves the final fusion step mostly static: after the two embeddings are computed, their interaction is often reduced to one algebraic similarity value.

This work asks whether a physically inspired interaction prior can improve frozen VLM matching. The motivating idea is to treat each modality as a concentration field rather than only a vector. Image and text representations are lifted into local fields, mixed by a learned gate, and evolved through a small number of advection, diffusion, and reaction updates. Advection models directed feature transport, diffusion models local smoothing and agreement, and reaction models nonlinear channel-wise transformation. The resulting module, FluidMix, is not a literal fluid simulator. It is a neural field-dynamic adapter inspired by continuous-depth models, liquid neural networks, and reaction-diffusion systems [@chen2018neuralode; @hasani2021liquid; @turing1952morphogenesis].

The central engineering lesson is residual anchoring. A naive FluidMix head that replaces CLIP cosine can destroy a strong pretrained ranking. Therefore, the main implementation adds a FluidMix residual to CLIP cosine and initializes the residual scale to zero. At initialization, the model is exactly CLIP. Training can only help if the learned residual correction becomes useful. A validation-calibrated variant further blends CLIP and residual scores using a coefficient selected only on validation data. This design makes the model safer, but the experiments show that it is not enough to guarantee improvement on general image-caption benchmarks.

The paper deliberately reports negative evidence. Earlier small-domain experiments suggested that calibrated residual FluidMix could improve CLIP on Pokemon-BLIP and small Flickr subsets. After adding COCO-1K/5K, full Flickr8k, larger Flickr30k, OpenCLIP, SigLIP, BLIP-ITM, and CLIP-LoRA controls, the claim boundary becomes sharper. FluidMix is promising as a residual adaptation mechanism in a domain-specific setting; the original pairwise-trained implementation fails on standard larger benchmarks. A subsequent hard-candidate retrieval objective fixes a major part of this failure in calibrated COCO top-k reranking, but the result is still a residual reranking result rather than a claim that FluidMix replaces pretrained VLMs. This distinction is important for a high-level journal submission: a scientifically credible paper should not turn a local positive result into an unsupported universal superiority claim.

The contributions are:

1. We formulate image-text matching as residual advection-diffusion-reaction mixing over learned modality concentration fields.
2. We provide a code-aligned mathematical derivation of the implemented FluidMix module, including gated field construction, discrete advection, grouped Laplacian diffusion, learnable reaction, residual scoring, validation calibration, and finite-step stability properties.
3. We compare FluidMix against CLIP zero-shot, OpenCLIP, SigLIP, BLIP-ITM, CLIP-LoRA, CLIP-free Pure FluidMix, and non-CLIP pretrained-backbone FluidMix baselines.
4. We extend evaluation from small image-caption subsets to COCO-1K, local COCO-5K, COCO Karpathy 5k, Flickr8k-full, and Flickr30k-large protocols with bidirectional R@1/R@5/R@10 retrieval, and add scalable one-caption and five-caption top-k reranking protocols with explicit candidate-recall ceilings.
5. We report both positive and negative findings, separating supported conclusions from unsupported claims about CLIP replacement, and show that aligning the residual objective with hard retrieval candidates changes the COCO failure mode into reproducible calibrated reranking gains.

## 2. Background and Related Work

### 2.1 Vision-Language Pretraining

CLIP learns aligned image and text encoders with a symmetric contrastive loss and enables zero-shot transfer through prompt-based matching [@radford2021clip]. ALIGN scales a similar idea with noisy image-text supervision [@jia2021align]. ALBEF, BLIP, and BLIP-2 add cross-modal fusion, bootstrapping, and frozen-component strategies [@li2021albef; @li2022blip; @li2023blip2]. OpenCLIP studies reproducible scaling of contrastive language-image learning and provides strong open checkpoints [@cherti2022openclip]. SigLIP replaces softmax contrastive learning with a sigmoid loss and is a strong modern embedding baseline [@zhai2023siglip]. These models set a high bar for any claim that a new fusion module improves VLM retrieval.

### 2.2 Multimodal Fusion

Classical multimodal fusion includes concatenation, gated fusion, bilinear pooling, tensor fusion, low-rank fusion, and cross-attention [@fukui2016mcb; @zadeh2017tfn; @liu2018lowrank]. Transformer VLMs such as ViLBERT and VisualBERT use cross-modal attention to model visual-linguistic interaction [@lu2019vilbert; @li2019visualbert]. FluidMix differs by lifting modality embeddings into local fields and applying spatial differential operators. It is best understood as a compact post-encoder adapter, not a replacement for high-capacity cross-modal pretraining.

### 2.3 Continuous-Time and Physics-Inspired Neural Models

Neural ODEs model hidden representations as continuous-depth dynamical systems [@chen2018neuralode]. Liquid time-constant networks use input-dependent dynamics for compact sequence modeling [@hasani2021liquid]. Physics-informed neural networks and universal differential equations show how differential-equation priors can be integrated with learning systems [@raissi2019pinn; @rackauckas2020ude]. Data-driven PDE discovery further shows that differential operators can provide useful structure [@rudy2017pde]. FluidMix borrows this language as an architectural prior for multimodal fusion; it does not infer a physical law and does not claim numerical fidelity to real fluid flow.

### 2.4 Parameter-Efficient Adaptation

LoRA injects low-rank trainable updates into frozen transformer projections, enabling efficient fine-tuning with a small number of parameters [@hu2021lora]. For VLM adaptation, LoRA is a critical baseline because it tests whether improvements come from the proposed fusion dynamics or simply from adding trainable parameters to CLIP. This paper therefore includes a local three-seed CLIP-LoRA baseline with rank 4 adapters on CLIP `q_proj` and `v_proj` modules.

## 3. Task Definition and Notation

The task is image-text matching and retrieval. Each item is a paired image-caption sample `(I_i,T_i)`. A model outputs a compatibility score

```math
s_{ij} = s(I_i,T_j).
```

For a test set of `N` paired samples, retrieval constructs a score matrix

```math
S \in \mathbb{R}^{N \times N}, \quad S_{ij}=s(I_i,T_j),
```

where the diagonal contains the ground-truth matches. Image-to-text retrieval ranks each row; text-to-image retrieval ranks each column. We report R@1/R@5/R@10. Binary image-text matching additionally evaluates positive diagonal pairs and sampled negative pairs using accuracy and AUC.

This is not a reinforcement learning, VLA, world-action-model, or robot-control task. The observation is an image-caption pair, the internal state is a learned concentration field, and the output is a compatibility score. There is no action, reward, environment transition, control frequency, or physical simulator step.

## 4. FluidMix Model Design

![Figure 1. FluidMix residual field-dynamic reranking pipeline. Frozen CLIP encoders provide the anchor score, while FluidMix evolves image/text concentration fields and contributes only a calibrated residual correction.](figures/fluidmix_architecture.png)

### 4.1 Frozen Encoder Interface

For CLIP-based variants, frozen encoders produce image and text embeddings:

```math
z_I = E_I(I) \in \mathbb{R}^{d}, \quad
z_T = E_T(T) \in \mathbb{R}^{d}.
```

For CLIP ViT-B/32, `d=512`. The zero-shot CLIP score is

```math
s_{\mathrm{CLIP}}(I,T)
= \gamma
\frac{z_I^\top z_T}{\lVert z_I\rVert_2\lVert z_T\rVert_2},
```

where the project implementation uses `gamma=1` for cached-feature adapter experiments.

### 4.2 Concentration Field Construction

FluidMix first maps each embedding into a global hidden vector:

```math
u_I = \mathrm{LN}(W_I^v z_I), \quad
u_T = \mathrm{LN}(W_T^v z_T),
\quad u_I,u_T \in \mathbb{R}^{H}.
```

It also maps each modality into a bounded field:

```math
C_I = \mathrm{reshape}(\tanh(W_I^f z_I)),
\quad
C_T = \mathrm{reshape}(\tanh(W_T^f z_T)),
```

```math
C_I,C_T \in \mathbb{R}^{H \times G \times G}.
```

The main configuration uses `H=128` and `G=4`, giving tensors of shape `[B,128,4,4]`.

### 4.3 Gated Initial Mixing

The two modality fields are mixed by a learned channel-wise gate:

```math
g = \sigma(W_g [u_I;u_T] + b_g), \quad g \in (0,1)^H.
```

The initial mixed field is

```math
C^0 = g \odot C_I + (1-g)\odot C_T.
```

This is a code-aligned description of the current implementation. The initial condition is not a simple concatenation and not a high-resolution physical grid; it is a compact learned field over which local operators can act.

### 4.4 Advection-Diffusion-Reaction Update

An idealized advection-diffusion-reaction equation can be written as

```math
\frac{\partial C}{\partial t}
= -v\cdot\nabla C + D\nabla^2 C + R(C),
```

where `C` is a concentration field, `v` is a velocity field, `D` is a diffusion coefficient, and `R(C)` is a reaction term. FluidMix implements a discrete neural version.

The velocity field is generated from the global image/text vectors:

```math
[v_x,v_y] = \tanh(W_v [u_I;u_T]+b_v),
\quad v_x,v_y \in \mathbb{R}^{H}.
```

Centered finite differences are approximated by circular shifts:

```math
\nabla_x C \approx \frac{1}{2}
\left(\mathrm{roll}_{x=-1}(C)-\mathrm{roll}_{x=+1}(C)\right),
```

```math
\nabla_y C \approx \frac{1}{2}
\left(\mathrm{roll}_{y=-1}(C)-\mathrm{roll}_{y=+1}(C)\right).
```

The advection term is

```math
A_{\mathrm{adv}}(C)
=-(v_x\odot\nabla_x C + v_y\odot\nabla_y C).
```

Diffusion uses a fixed five-point Laplacian kernel applied independently to each channel:

```math
K_L =
\begin{bmatrix}
0 & 1 & 0\\
1 & -4 & 1\\
0 & 1 & 0
\end{bmatrix},
\quad
A_{\mathrm{diff}}(C)
=D\cdot\mathrm{Conv2D}_{group}(C,K_L).
```

The reaction term is a learnable `1 x 1` convolution:

```math
A_{\mathrm{react}}(C)
=\rho\cdot\tanh(\mathrm{Conv}_{1\times 1}(C)).
```

The explicit Euler update is

```math
C^{k+1}
= C^k+\Delta t
\left[
A_{\mathrm{adv}}(C^k)
+A_{\mathrm{diff}}(C^k)
+A_{\mathrm{react}}(C^k)
\right].
```

The default solver uses `steps=4`, `dt=0.2`, `diffusion=0.03`, `reaction=0.5`, and `field_grid=4`.

### 4.5 Scoring, Residual Anchoring, and Calibration

After `K` solver steps, the field is average-pooled and normalized:

```math
h = \mathrm{LN}\left(
\frac{1}{G^2}\sum_{x=1}^{G}\sum_{y=1}^{G} C^K_{:,x,y}
\right).
```

An MLP produces a FluidMix matching logit:

```math
r_{\mathrm{FM}}
= W_2\mathrm{Dropout}(\mathrm{GELU}(W_1h+b_1))+b_2.
```

Ordinary FluidMix uses `r_FM` directly. Residual FluidMix uses

```math
s_{\mathrm{res}}(I,T)
=s_{\mathrm{CLIP}}(I,T)+\alpha r_{\mathrm{FM}}(I,T).
```

The scalar `alpha` is initialized to zero, so

```math
s_{\mathrm{res}}(I,T)\vert_{\alpha=0}
=s_{\mathrm{CLIP}}(I,T).
```

Validation-calibrated residual FluidMix chooses a blending coefficient on validation data:

```math
s_{\mathrm{cal}}(I,T)
=(1-\lambda)s_{\mathrm{CLIP}}(I,T)
+\lambda s_{\mathrm{res}}(I,T).
```

The selected `lambda` is then fixed for test evaluation.

### 4.6 Mathematical Properties and Claim Boundary

This section makes explicit what the FluidMix construction guarantees and what it does not guarantee. The guarantees below are properties of the implemented finite-step adapter. They do not imply that FluidMix is universally better than CLIP; they justify why the residual and calibrated form is a controlled correction rather than an unconstrained replacement.

Let `V = R^{H x G x G}` be the field space with Frobenius norm `||.||_F`. For one image-text pair, define the update map

```math
\Phi(C)=C+\Delta t F(C),
```

where

```math
F(C)=A_{\mathrm{adv}}(C)+A_{\mathrm{diff}}(C)+A_{\mathrm{react}}(C).
```

**Proposition 1 (finite-step Lipschitz stability).** Assume that the reaction convolution has finite spectral norm `||W_R||_2`. Then one FluidMix Euler step is Lipschitz in `C`, and after `K` steps the perturbation growth is bounded by a finite product.

*Proof.* The implemented gradients use channel-wise central differences with circular shifts, so their induced operator norms satisfy

```math
\|D_x\|_2 \le 1,\qquad \|D_y\|_2 \le 1.
```

Because `v_x` and `v_y` are generated by `tanh`, each channel velocity is bounded by one. Hence

```math
\|A_{\mathrm{adv}}(C)-A_{\mathrm{adv}}(C')\|_F
\le 2\|C-C'\|_F.
```

For the five-point Laplacian kernel, Young's convolution inequality gives the conservative bound

```math
\|A_{\mathrm{diff}}(C)-A_{\mathrm{diff}}(C')\|_F
\le 8D\|C-C'\|_F,
```

because the absolute kernel sum is `8D`. If the `1 x 1` reaction convolution has spectral norm `||W_R||_2`, then `tanh` is 1-Lipschitz and

```math
\|A_{\mathrm{react}}(C)-A_{\mathrm{react}}(C')\|_F
\le \rho \|W_R\|_2\|C-C'\|_F.
```

Therefore the complete vector field is Lipschitz with

```math
L_F \le 2+8D+\rho\|W_R\|_2,
```

and the explicit Euler map is finite-step Lipschitz:

```math
\|\Phi(C)-\Phi(C')\|_F
\le (1+\Delta t L_F)\|C-C'\|_F.
```

After `K` steps,

```math
\|C^K-\tilde C^K\|_F
\le (1+\Delta t L_F)^K\|C^0-\tilde C^0\|_F.
```

This proves the finite-horizon statement. For the small fixed `K=4` used in the main experiments, perturbations in the initial concentration field cannot be amplified without bound. It is not an asymptotic stability theorem for arbitrarily long integration.

**Proposition 2 (finite-step boundedness).** For any finite number of solver steps `K`, the implemented FluidMix field remains bounded.

The initial fields are bounded because they are produced by `tanh`, and the gate is a convex mixture. Thus `||C^0||_\infty <= 1`. Let `d` be the number of field channels and let `H,W` be the spatial grid size. Since the reaction term is bounded by `rho` per field element and the linear advection-diffusion part has norm bounded by `2+8D`,

```math
\|C^{k+1}\|_F
\le
\left(1+\Delta t(2+8D)\right)\|C^k\|_F
+ \Delta t\rho\sqrt{dHW}.
```

By induction, the finite-step representation is bounded for every finite `K`. This explains why the adapter is numerically well-defined, but it does not by itself explain retrieval improvement.

**Proposition 3 (residual ranking preservation at initialization).** At initialization, residual FluidMix exactly preserves every CLIP pairwise ranking. More generally, pairwise order is preserved whenever the residual perturbation is smaller than the CLIP margin.

The residual design gives a separate ranking-safety property. For any query `q`, let the CLIP margin between candidate `a` and candidate `b` be

```math
m_{ab}=s_{\mathrm{CLIP}}(q,a)-s_{\mathrm{CLIP}}(q,b).
```

The residual score changes this margin by

```math
m^{\mathrm{res}}_{ab}
=m_{ab}+\alpha\left(r_{\mathrm{FM}}(q,a)-r_{\mathrm{FM}}(q,b)\right).
```

Therefore, if

```math
|\alpha|\cdot
|r_{\mathrm{FM}}(q,a)-r_{\mathrm{FM}}(q,b)|
< |m_{ab}|,
```

then the pairwise order between `a` and `b` is preserved. In particular, at initialization `alpha=0`, all CLIP pairwise orders and all CLIP retrieval metrics are exactly preserved. This is the formal reason ordinary FluidMix and residual FluidMix should not be interpreted as the same hypothesis: ordinary FluidMix discards the pretrained ranking, while residual FluidMix starts from it.

**Proposition 4 (finite-grid validation calibration).** Selecting `lambda` from a finite validation grid has a standard uniform-convergence bound and admits `lambda=0` as a safe fallback.

Validation calibration can also be stated as a finite model-selection problem. Let `Lambda` be the grid of candidate mixing weights and let `M(lambda)` be any bounded validation metric in `[0,1]`, such as R@1 or a normalized sum of retrieval recalls. For `n` validation queries, Hoeffding's inequality [@hoeffding1963probability] and a union bound imply that, with probability at least `1-delta`,

```math
\sup_{\lambda\in\Lambda}
|\widehat M_n(\lambda)-M(\lambda)|
\le
\sqrt{\frac{\log(2|\Lambda|/\delta)}{2n}}.
```

If `hat lambda` maximizes the empirical validation metric over the grid, then its population metric is within twice this deviation term of the best grid element. This does not prove that a useful nonzero residual exists; it proves that selecting `lambda` on validation is a controlled finite-class selection procedure and that `lambda=0` is a legitimate outcome when the residual is harmful.

**Proposition 5 (top-k reranking ceiling).** A reranker restricted to a first-stage candidate set cannot exceed the candidate recall upper bound.

Finally, for large retrieval sets we use a two-stage top-k reranking protocol. Let `K_q` be the CLIP candidate set for query `q`. Any reranker restricted to `K_q` satisfies

```math
\mathbf{1}\{\mathrm{rank}_{\mathrm{rerank}}(q,y_q)\le r\}
\le
\mathbf{1}\{y_q\in K_q\}.
```

Thus top-k reranking recall is upper-bounded by candidate recall. Reporting `candidate_recall@topk` is therefore required for an honest scalable COCO-5K FluidMix reranking claim.

## 5. Optimization

The adapter training objective combines binary image-text matching and optional symmetric retrieval contrastive loss. For labeled pairs `(I,T,y)`,

```math
L_{\mathrm{BCE}}
= -y\log\sigma(s) -(1-y)\log(1-\sigma(s)).
```

For a batch score matrix `S`, the symmetric retrieval loss is

```math
L_{\mathrm{ITC}}
=\frac{1}{2}
\left[
\mathrm{CE}(S/\tau,\mathrm{diag})
+\mathrm{CE}(S^\top/\tau,\mathrm{diag})
\right].
```

The total adapter objective is

```math
L=L_{\mathrm{BCE}}+\beta L_{\mathrm{ITC}}.
```

The contrastive term is used because binary matching and retrieval optimize different objects. Binary BCE only constrains sampled positive and negative pairs. Retrieval requires the diagonal score to dominate all non-diagonal scores in a row or column. For one image-to-text row `i`, the contrastive loss is

```math
\ell_i^{I\rightarrow T}
=-\log
\frac{\exp(S_{ii}/\tau)}
{\sum_j \exp(S_{ij}/\tau)}
=\log\left(1+\sum_{j\ne i}
\exp((S_{ij}-S_{ii})/\tau)\right).
```

Let

```math
\Delta_i=\max_{j\ne i}(S_{ij}-S_{ii})
```

be the largest violation margin for row `i`. Then

```math
\ell_i^{I\rightarrow T}
\ge \log\left(1+\exp(\Delta_i/\tau)\right).
```

Equivalently, reducing `ell_i` pushes the largest nonmatching caption score below the matching caption score. The text-to-image term gives the same pressure over columns. This explains why `L_ITC` is a direct surrogate for retrieval ranking, while the hard-negative BCE term focuses on local pair discrimination. The ablation results show that this pressure is dataset-dependent: it helps Pokemon-BLIP but can disturb strong CLIP rankings on general Flickr subsets.

Motivated by the COCO failure of the BCE-dominant recipe, we also introduce a retrieval-first residual variant. For an in-batch score matrix from residual FluidMix, `S_res`, and the corresponding frozen CLIP matrix, `S_clip`, the objective is

```math
L_{\mathrm{RF}}
=
\gamma L_{\mathrm{ITC}}(S_{\mathrm{res}})
+\eta\lVert S_{\mathrm{res}}-S_{\mathrm{CLIP}}\rVert_F^2
+\zeta\alpha^2.
```

The first term directly optimizes bidirectional retrieval. The second term is a residual-geometry regularizer that discourages the adapter from destroying CLIP's global score geometry. The last term limits the residual scale. This objective is implemented in `scripts/run_retrieval_first_residual.py` and is evaluated as a targeted attempt to close the COCO ranking failure.

The stronger candidate-retrieval variant replaces in-batch negatives with frozen-CLIP hard candidates, matching the final top-k reranking protocol more directly. For each image query `i`, let `H_i^T` be the `h` highest-scoring nonmatching captions under frozen CLIP, and define the candidate set

```math
\mathcal{C}_i^T=\{i\}\cup H_i^T.
```

For each text query `j`, define `H_j^I` and

```math
\mathcal{C}_j^I=\{j\}\cup H_j^I
```

analogously over candidate images. The bidirectional hard-candidate retrieval loss is

```math
L_{\mathrm{HC}}
=
\frac{1}{2B}
\sum_{i=1}^{B}
\left[
-\log
\frac{\exp(s_{\mathrm{res}}(I_i,T_i)/\tau)}
{\sum_{j\in\mathcal{C}_i^T}\exp(s_{\mathrm{res}}(I_i,T_j)/\tau)}
+
-\log
\frac{\exp(s_{\mathrm{res}}(I_i,T_i)/\tau)}
{\sum_{k\in\mathcal{C}_i^I}\exp(s_{\mathrm{res}}(I_k,T_i)/\tau)}
\right].
```

The optimized objective is

```math
L_{\mathrm{cand}}
=
\gamma L_{\mathrm{HC}}
+
\eta
\mathbb{E}_{(i,j)\in\mathcal{P}_{\mathrm{cand}}}
\left(s_{\mathrm{res}}(I_i,T_j)-s_{\mathrm{CLIP}}(I_i,T_j)\right)^2
+
\zeta\alpha^2,
```

where `P_cand` is the union of image-to-text and text-to-image candidate pairs. The cross-entropy term asks the residual model to separate the positive pair from CLIP-confusable alternatives; the geometry term keeps the score surface near CLIP unless the candidate loss supplies local ranking evidence. This objective is implemented in `scripts/run_candidate_retrieval_residual.py` with `h=64`, `tau=0.07`, `gamma=1`, `eta=10`, `zeta=0.1`, and validation-selected `lambda` over the same finite grid used by the top-k reranker.

This hard-candidate objective is a local ranking surrogate for the actual reranking decision region. For one image query, define

```math
\Delta_i^{\mathrm{HC}}
=
\max_{j\in\mathcal{C}_i^T\setminus\{i\}}
\left(s_{\mathrm{res}}(I_i,T_j)-s_{\mathrm{res}}(I_i,T_i)\right).
```

The image-to-text hard-candidate loss can be rewritten as

```math
\ell_i^{\mathrm{HC}}
=
\log\left(1+
\sum_{j\in\mathcal{C}_i^T\setminus\{i\}}
\exp\left((s_{\mathrm{res}}(I_i,T_j)-s_{\mathrm{res}}(I_i,T_i))/\tau\right)
\right).
```

Therefore

```math
\ell_i^{\mathrm{HC}}
\ge
\log\left(1+\exp(\Delta_i^{\mathrm{HC}}/\tau)\right).
```

If optimization makes `ell_i^HC <= epsilon`, then

```math
\Delta_i^{\mathrm{HC}}
\le
\tau\log\left(\exp(\epsilon)-1\right).
```

Thus reducing the hard-candidate loss directly upper-bounds the largest positive-vs-hard-negative score violation inside the CLIP candidate neighborhood. This is the mathematical reason it is better aligned with top-k reranking than the original BCE recipe: it spends capacity on the high-score confusions that can actually change the final rank, while the geometry term and residual calibration prevent unconstrained movement away from CLIP.

Hard negatives are selected using high-scoring nonmatching captions under frozen CLIP similarity. The CLIP-LoRA baseline uses the standard in-batch symmetric CLIP contrastive cross entropy:

```math
L_{\mathrm{LoRA}}
=\frac{1}{2}
\left[
\mathrm{CE}(\ell_{I\rightarrow T},\mathrm{diag})
+\mathrm{CE}(\ell_{T\rightarrow I},\mathrm{diag})
\right],
```

where LoRA updates only the low-rank adapter parameters in CLIP attention projections.

## 6. Experimental Protocol

### 6.1 Datasets

| Dataset protocol | Train | Val | Test | Notes |
| --- | ---: | ---: | ---: | --- |
| Pokemon-BLIP | 240 | 80 | 80 | Domain-specific cartoon image-caption data |
| Flickr8k subset | 1000 | 200 | 200 | Earlier local subset for broad ablations |
| Flickr30k subset | 1000 | 200 | 200 | Earlier local subset for broad ablations |
| COCO-1K | 5000 | 1000 | 1000 | COCO 2014 local protocol, one caption per image |
| COCO-5K | 10000 | 1000 | 4954 | Local `5k.part` list; 4954 images matched available captions |
| COCO-Karpathy | 10000 | 5000 | 5000 | COCO 2014 Karpathy split [@karpathy2015deep]; five-caption evaluation uses up to five references per image |
| Flickr8k-full | 6000 | 1000 | 1000 | Official Flickr8k split files |
| Flickr30k-large | 10000 | 1000 | 1000 | Local deterministic split from Flickr30k archive |

The main COCO and Flickr protocol tables use one caption per image to form a square image-caption retrieval matrix. This differs from the common five-caption COCO/Flickr evaluation protocol and should be treated as a local controlled benchmark rather than a leaderboard-comparable result. COCO-5K has 4954 test items because the local `5k.part` list did not fully match the available caption annotations. To reduce this external-validity gap, we additionally materialize the standard COCO Karpathy split with 5000 validation images and 5000 test images; missing images from the local zip are fetched into a supplemental image directory while keeping the Karpathy split assignments fixed.

### 6.2 Compared Models

The evaluated models are:

- CLIP ViT-B/32 zero-shot;
- OpenCLIP ViT-B/32 LAION2B;
- SigLIP base patch16-224;
- BLIP-ITM base COCO as a cross-encoder retrieval baseline on a 200-item COCO-1K subset;
- CLIP-LoRA with rank 4 adapters on `q_proj` and `v_proj`, trained for 2 epochs and repeated with seeds `7,13,21`;
- ordinary FluidMix;
- residual FluidMix;
- validation-calibrated residual FluidMix;
- retrieval-first geometry-regularized residual FluidMix;
- hard-candidate retrieval residual FluidMix;
- Pure FluidMix without CLIP;
- ResNet50+MiniLM+FluidMix without CLIP.

The local weight cache contains OpenCLIP, SigLIP, and BLIP-ITM. BLIP-2 weights are not present in the verified weight directory, so BLIP-2 is not reported as a completed result.

### 6.3 Metrics, Seeds, and Hardware

Retrieval reports image-to-text and text-to-image R@1/R@5/R@10. Binary matching reports accuracy and AUC with the threshold selected on validation data. Main adapter runs use seed `13`; deterministic split seed is `42`; earlier calibrated residual sweeps use seeds `7,13,23,42,99`; the hard-candidate retrieval residual variant, five-caption COCO-Karpathy check, and CLIP-LoRA baseline are repeated with seeds `7,13,21`. Experiments were run locally with PyTorch, Transformers, PEFT, and an NVIDIA GeForce RTX 3060 Laptop GPU.

For large-set FluidMix evaluation, we additionally provide a scalable top-k reranking protocol. A first-stage CLIP retriever selects the top-k candidate captions or images for each query. Residual FluidMix is then evaluated only inside this candidate set, and the final calibrated reranking score uses the validation-selected coefficient

```math
s_{\mathrm{rerank}}=(1-\lambda)s_{\mathrm{CLIP}}+\lambda s_{\mathrm{res}}.
```

This reduces adapter scoring from `N^2` pairs to `N k` pairs per retrieval direction. Because this protocol cannot recover a target excluded by the CLIP candidate set, reranking results must be reported together with `candidate_recall@topk`. The implementation is `scripts/run_topk_fluidmix_rerank.py`.

## 7. Results

### 7.1 Earlier Small-Domain and Subset Results

| Dataset | Model | Uses CLIP | Accuracy | AUC | i2t R@1 | i2t R@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Pokemon-BLIP | CLIP zero-shot | yes | 0.8000 | 0.8683 | 0.1875 | 0.5625 |
| Pokemon-BLIP | ordinary FluidMix | frozen embeddings | 0.4813 | 0.5022 | 0.0250 | 0.0375 |
| Pokemon-BLIP | residual FluidMix | frozen CLIP + residual | 0.8187 | 0.8775 | 0.3750 | 0.7125 |
| Pokemon-BLIP | calibrated residual FluidMix | frozen CLIP + calibration | 0.8187 | 0.8752 | 0.4000 | 0.7250 |
| Pokemon-BLIP | Pure FluidMix | no | 0.5250 | 0.5830 | 0.0250 | 0.1125 |
| Pokemon-BLIP | ResNet50+MiniLM+FluidMix | no | 0.6813 | 0.7516 | 0.0750 | 0.2500 |
| Flickr8k subset | CLIP zero-shot | yes | 0.9725 | 0.9899 | 0.7600 | 0.9650 |
| Flickr8k subset | calibrated residual FluidMix | frozen CLIP + calibration | 0.9750 | 0.9919 | 0.7800 | 0.9600 |
| Flickr30k subset | CLIP zero-shot | yes | 0.9750 | 0.9977 | 0.8900 | 0.9900 |
| Flickr30k subset | calibrated residual FluidMix | frozen CLIP + calibration | 0.9800 | 0.9968 | 0.8950 | 0.9950 |

Pokemon-BLIP remains the strongest positive FluidMix result. Paired bootstrap for calibrated residual FluidMix versus CLIP gives an i2t R@1 delta of `+0.2125` with 95% CI `[+0.1125,+0.3125]`, and an i2t R@5 delta of `+0.1625` with 95% CI `[+0.0750,+0.2625]`. The Flickr subset gains are small and not sufficient for a broad superiority claim.

### 7.2 Foundation VLM Benchmark

| Dataset | Model | Samples | i2t R@1 | i2t R@5 | t2i R@1 | t2i R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COCO-1K | CLIP-B/32 | 1000 | 0.538 | 0.813 | 0.491 | 0.781 |
| COCO-1K | OpenCLIP-B/32 LAION2B | 1000 | 0.608 | 0.863 | 0.571 | 0.836 |
| COCO-1K | SigLIP-B/16 | 1000 | 0.645 | 0.903 | 0.642 | 0.879 |
| COCO-1K | BLIP-ITM COCO | 200 | 0.910 | 1.000 | 0.910 | 0.990 |
| COCO-5K | CLIP-B/32 | 4954 | 0.3387 | 0.5856 | 0.2951 | 0.5394 |
| COCO-5K | OpenCLIP-B/32 LAION2B | 4954 | 0.3902 | 0.6585 | 0.3708 | 0.6427 |
| COCO-5K | SigLIP-B/16 | 4954 | 0.4524 | 0.7063 | 0.4522 | 0.7006 |
| Flickr8k-full | CLIP-B/32 | 1000 | 0.576 | 0.835 | 0.547 | 0.817 |
| Flickr8k-full | OpenCLIP-B/32 LAION2B | 1000 | 0.644 | 0.878 | 0.626 | 0.860 |
| Flickr8k-full | SigLIP-B/16 | 1000 | 0.723 | 0.944 | 0.717 | 0.937 |
| Flickr30k-large | CLIP-B/32 | 1000 | 0.715 | 0.906 | 0.683 | 0.895 |
| Flickr30k-large | OpenCLIP-B/32 LAION2B | 1000 | 0.779 | 0.949 | 0.765 | 0.929 |
| Flickr30k-large | SigLIP-B/16 | 1000 | 0.843 | 0.963 | 0.836 | 0.955 |

![Figure 2. Foundation embedding-model image-to-text R@1 across local benchmark protocols. SigLIP is the strongest downloaded embedding baseline, which bounds how broadly FluidMix can claim improvement.](../results/q1_benchmark_synthesis/foundation_i2t_r1.png)

SigLIP is the strongest embedding baseline among the downloaded models. BLIP-ITM performs very strongly on the 200-item COCO-1K subset, but it is a cross-encoder and therefore not directly comparable to full 1K/5K embedding retrieval in compute cost.

### 7.3 FluidMix on Standard Benchmarks

| Dataset | Model | Accuracy | AUC | i2t R@1 | i2t R@5 | Selected lambda |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COCO-1K | CLIP zero-shot | 0.9760 | 0.9955 | 0.538 | 0.813 | - |
| COCO-1K | raw residual FluidMix | 0.9455 | 0.9813 | 0.213 | 0.526 | - |
| COCO-1K | calibrated residual FluidMix | 0.9760 | 0.9955 | 0.538 | 0.813 | 0.0 |
| Flickr8k-full | CLIP zero-shot | 0.9710 | 0.9938 | 0.576 | 0.835 | - |
| Flickr8k-full | raw residual FluidMix | 0.9185 | 0.9654 | 0.081 | 0.283 | - |
| Flickr8k-full | calibrated residual FluidMix | 0.9710 | 0.9938 | 0.576 | 0.835 | 0.0 |
| Flickr30k-large | CLIP zero-shot | 0.9720 | 0.9970 | 0.715 | 0.906 | - |
| Flickr30k-large | raw residual FluidMix | 0.9275 | 0.9726 | 0.139 | 0.399 | - |
| Flickr30k-large | calibrated residual FluidMix | 0.9720 | 0.9970 | 0.715 | 0.906 | 0.0 |

![Figure 3. Standard all-pairs image-to-text R@1 result. The raw residual recipe damages CLIP rankings on COCO/Flickr, while validation calibration safely returns to CLIP with lambda=0.](../results/q1_benchmark_synthesis/fluidmix_standard_i2t_r1.png)

This is the most important negative result. On larger standard benchmarks, raw residual FluidMix severely degrades retrieval. The validation-calibrated model avoids harm by selecting `lambda=0`, which exactly reverts to CLIP. Therefore, the standard benchmark results do not show FluidMix outperforming CLIP.

### 7.4 CLIP-LoRA Baseline

The CLIP-LoRA baseline trains rank-4 adapters on CLIP `q_proj` and `v_proj` modules for two epochs, with `245760` trainable parameters. We repeat it with seeds `7,13,21`; the table reports mean `+-` sample standard deviation.

| Dataset | Direction | CLIP R@1/R@5/R@10 | CLIP-LoRA R@1/R@5/R@10 | Delta R@1/R@5/R@10 | Seeds |
| --- | --- | --- | --- | --- | ---: |
| COCO-1K | i2t | 0.5380 / 0.8140 / 0.9000 | 0.5490+-0.0010 / 0.8217+-0.0021 / 0.9027+-0.0012 | +0.0110 / +0.0077 / +0.0027 | 3 |
| COCO-1K | t2i | 0.4900 / 0.7820 / 0.8810 | 0.5007+-0.0006 / 0.7873+-0.0012 / 0.8857+-0.0015 | +0.0107 / +0.0053 / +0.0047 | 3 |
| Flickr8k-full | i2t | 0.5760 / 0.8350 / 0.9140 | 0.6060+-0.0026 / 0.8620+-0.0026 / 0.9330+-0.0020 | +0.0300 / +0.0270 / +0.0190 | 3 |
| Flickr8k-full | t2i | 0.5450 / 0.8180 / 0.9020 | 0.5843+-0.0025 / 0.8477+-0.0021 / 0.9120+-0.0017 | +0.0393 / +0.0297 / +0.0100 | 3 |
| Flickr30k-large | i2t | 0.7140 / 0.9050 / 0.9420 | 0.7430+-0.0026 / 0.9303+-0.0012 / 0.9607+-0.0006 | +0.0290 / +0.0253 / +0.0187 | 3 |
| Flickr30k-large | t2i | 0.6840 / 0.8950 / 0.9430 | 0.7157+-0.0025 / 0.9190+-0.0000 / 0.9550+-0.0010 | +0.0317 / +0.0240 / +0.0120 | 3 |

![Figure 4. CLIP-LoRA matched baseline R@1 deltas over zero-shot CLIP. The PEFT control is consistently positive, so FluidMix is positioned as a different residual reranking mechanism rather than as the only useful adapter.](figures/clip_lora_r1_delta.png)

The LoRA baseline improves zero-shot CLIP on all three tested standard protocols: `54/54` per-seed recall deltas are positive in `results/q1_clip_lora_multiseed/clip_lora_multiseed_report.md`. This does not make LoRA a full replacement for large-scale pretraining, but it shows that direct parameter-efficient CLIP adaptation is a strong standard-benchmark baseline that any FluidMix claim must respect.

### 7.5 Ablation and Complexity Summary

Earlier ablations show that residual anchoring is more important than the specific nonlinear head. Ordinary FluidMix, concat, gated, cross-attention, and liquid fusion variants tend to degrade when they replace CLIP cosine directly. Dynamics ablations also show dataset dependence: Pokemon-BLIP benefits from hard negatives and contrastive retrieval training, while Flickr subsets prefer weaker contrastive settings. Adapter-only latency on CUDA batch 256 is approximately 5.43 ms for residual FluidMix, compared with 1.32 ms for gated fusion and 2.07 ms for cross-attention. This overhead is small for cached-feature reranking but not negligible for large all-pairs retrieval.

### 7.6 Scalable Top-k Reranking Check

To verify that the standard-benchmark negative result is not only an artifact of expensive all-pairs evaluation, we ran the new top-k reranking protocol on COCO-1K and COCO-5K with CLIP top-200 candidates. The candidate recall upper bound is effectively saturated for both protocols, so the reranker has enough candidate coverage to improve the top ranks if the residual correction is useful.

| Dataset | Model | Direction | Top-k | Candidate recall | R@1 | R@5 | R@10 | Selected lambda |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO-1K | CLIP zero-shot | i2t | 200 | 1.000 | 0.538 | 0.813 | 0.900 | - |
| COCO-1K | residual FluidMix rerank | i2t | 200 | 1.000 | 0.217 | 0.538 | 0.717 | - |
| COCO-1K | calibrated residual rerank | i2t | 200 | 1.000 | 0.538 | 0.813 | 0.900 | 0.0 |
| COCO-1K | CLIP zero-shot | t2i | 200 | 0.997 | 0.491 | 0.781 | 0.881 | - |
| COCO-1K | residual FluidMix rerank | t2i | 200 | 0.997 | 0.298 | 0.606 | 0.754 | - |
| COCO-1K | calibrated residual rerank | t2i | 200 | 0.997 | 0.491 | 0.781 | 0.881 | 0.0 |
| COCO-5K | CLIP zero-shot | i2t | 200 | 0.9925 | 0.3387 | 0.5856 | 0.7000 | - |
| COCO-5K | residual FluidMix rerank | i2t | 200 | 0.9925 | 0.0832 | 0.2553 | 0.3862 | - |
| COCO-5K | calibrated residual rerank | i2t | 200 | 0.9925 | 0.3387 | 0.5856 | 0.7000 | 0.0 |
| COCO-5K | CLIP zero-shot | t2i | 200 | 0.9883 | 0.2951 | 0.5394 | 0.6583 | - |
| COCO-5K | residual FluidMix rerank | t2i | 200 | 0.9883 | 0.1163 | 0.3157 | 0.4526 | - |
| COCO-5K | calibrated residual rerank | t2i | 200 | 0.9883 | 0.2951 | 0.5394 | 0.6583 | 0.0 |

This scalable check confirms the same boundary as the all-pairs standard benchmark: the current residual correction is not a useful COCO reranker, and validation calibration safely rejects it. We also ran a signed calibration sweep over `lambda in [-2,2]`; validation still selected `lambda=0`, indicating that the failure is not merely a wrong residual sign. The value of this experiment is methodological rather than positive: future COCO-5K runs can use top-k reranking while honestly reporting the candidate-recall ceiling.

### 7.7 Retrieval-first Regularized Residual Variant

The previous negative results suggest that pairwise BCE-dominant training is misaligned with retrieval. We therefore trained a retrieval-first residual variant on COCO-1K cached features using in-batch bidirectional contrastive loss plus a score-geometry regularizer toward CLIP. This variant does not fully solve COCO retrieval, but it changes the failure mode: image-to-text remains slightly below CLIP, while text-to-image top-k retrieval improves after validation calibration.

| Dataset | Model | Direction | Top-k | R@1 | R@5 | R@10 | Delta R@5 | Delta R@10 | Selected lambda |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO-1K | CLIP zero-shot | i2t | 200 | 0.538 | 0.813 | 0.900 | - | - | - |
| COCO-1K | retrieval-first calibrated residual | i2t | 200 | 0.526 | 0.807 | 0.904 | -0.006 | +0.004 | 0.4 |
| COCO-1K | CLIP zero-shot | t2i | 200 | 0.491 | 0.781 | 0.881 | - | - | - |
| COCO-1K | retrieval-first calibrated residual | t2i | 200 | 0.498 | 0.797 | 0.893 | +0.016 | +0.012 | 0.4 |
| COCO-5K | CLIP zero-shot | i2t | 200 | 0.3387 | 0.5856 | 0.7000 | - | - | - |
| COCO-5K | retrieval-first calibrated residual | i2t | 200 | 0.3361 | 0.5892 | 0.6970 | +0.0036 | -0.0030 | 0.4 |
| COCO-5K | CLIP zero-shot | t2i | 200 | 0.2951 | 0.5394 | 0.6583 | - | - | - |
| COCO-5K | retrieval-first calibrated residual | t2i | 200 | 0.3010 | 0.5543 | 0.6696 | +0.0149 | +0.0113 | 0.4 |

Paired bootstrap supports the COCO-5K text-to-image improvements: R@5 delta `+0.0149` with 95% CI `[+0.0079,+0.0212]`, and R@10 delta `+0.0113` with 95% CI `[+0.0052,+0.0172]`. COCO-1K text-to-image R@5 is also positive with CI `[+0.002,+0.031]`. These are limited but important results: a retrieval-first objective with geometry regularization can recover some standard-benchmark benefit, while the asymmetric degradation on image-to-text shows that the method is still not a universal CLIP improvement.

### 7.8 Hard-candidate Retrieval Residual Variant

The in-batch retrieval-first variant still uses negatives determined by minibatch composition. We therefore trained the hard-candidate retrieval residual variant defined in Section 5, using the top 64 frozen-CLIP nonmatches per training query and the same CLIP score-geometry regularizer. The adapter is trained on COCO-1K train cached CLIP features. For evaluation, CLIP first selects top-200 candidates, the residual adapter scores only those candidates, and `lambda` is selected on the COCO-1K validation split. All three seeds select `lambda=0.2`.

The raw residual scores remain poor and are not used as final predictions. The improvement appears only after validation calibration, which is consistent with the residual-correction interpretation: the learned field-dynamic score is useful as a small local correction to CLIP, not as a standalone ranker.

| Dataset | Direction | CLIP R@1/R@5/R@10 | Candidate residual R@1/R@5/R@10, mean +- std | Mean delta R@1/R@5/R@10 | Candidate recall | Seeds |
| --- | --- | --- | --- | --- | ---: | ---: |
| COCO-1K | i2t | 0.538 / 0.813 / 0.900 | 0.5707+-0.0070 / 0.8300+-0.0000 / 0.9127+-0.0042 | +0.0327 / +0.0170 / +0.0127 | 1.0000 | 3 |
| COCO-1K | t2i | 0.491 / 0.781 / 0.881 | 0.5393+-0.0050 / 0.8180+-0.0056 / 0.9043+-0.0068 | +0.0483 / +0.0370 / +0.0233 | 0.9970 | 3 |
| COCO-5K | i2t | 0.3387 / 0.5856 / 0.7000 | 0.3617+-0.0032 / 0.6172+-0.0047 / 0.7310+-0.0026 | +0.0230 / +0.0316 / +0.0310 | 0.9925 | 3 |
| COCO-5K | t2i | 0.2951 / 0.5394 / 0.6583 | 0.3334+-0.0035 / 0.5992+-0.0039 / 0.7161+-0.0036 | +0.0383 / +0.0599 / +0.0578 | 0.9883 | 3 |

Paired bootstrap is computed independently for each seed. For COCO-5K, every direction and metric has a positive worst-case 95% CI lower bound across the three seeds: i2t R@1/R@5/R@10 lower bounds are `+0.0123/+0.0176/+0.0206`, and t2i R@1/R@5/R@10 lower bounds are `+0.0256/+0.0476/+0.0454`. COCO-1K is also positive for all mean deltas; its weakest case is i2t R@10 for seed 21, whose CI lower bound is `-0.0040`, so the COCO-1K R@10 result should be treated as weaker than the COCO-5K result. The full aggregation is in `results/q1_candidate_retrieval_multiseed/candidate_retrieval_multiseed_report.md`.

We also ran the same hard-candidate protocol on Flickr8k-full and Flickr30k-large with seeds `7,13,21`. All mean deltas are positive.

| Dataset | Direction | CLIP R@1/R@5/R@10 | Candidate residual R@1/R@5/R@10, mean +- std | Mean delta R@1/R@5/R@10 | Worst 95% CI lower bounds |
| --- | --- | --- | --- | --- | --- |
| Flickr8k-full | i2t | 0.576 / 0.835 / 0.914 | 0.6153+-0.0101 / 0.8673+-0.0040 / 0.9243+-0.0038 | +0.0393 / +0.0323 / +0.0103 | +0.010 / +0.013 / -0.007 |
| Flickr8k-full | t2i | 0.547 / 0.817 / 0.902 | 0.5917+-0.0067 / 0.8630+-0.0020 / 0.9263+-0.0049 | +0.0447 / +0.0460 / +0.0243 | +0.016 / +0.027 / +0.006 |
| Flickr30k-large | i2t | 0.715 / 0.906 / 0.942 | 0.7493+-0.0057 / 0.9277+-0.0006 / 0.9563+-0.0032 | +0.0343 / +0.0217 / +0.0143 | +0.009 / +0.009 / +0.002 |
| Flickr30k-large | t2i | 0.683 / 0.895 / 0.944 | 0.7593+-0.0070 / 0.9247+-0.0031 / 0.9580+-0.0036 | +0.0763 / +0.0297 / +0.0140 | +0.048 / +0.013 / -0.001 |

![Figure 5. Hard-candidate retrieval residual mean recall deltas over CLIP across three seeds. All mean deltas are positive across the evaluated COCO/Flickr one-caption protocols.](../results/q1_candidate_retrieval_multiseed/candidate_delta_grid.png)

![Figure 6. Worst paired-bootstrap lower bounds across seeds for the hard-candidate retrieval residual. Most weak cases occur at already-high recall levels.](../results/q1_candidate_retrieval_multiseed/candidate_bootstrap_floor.png)

The weak negative lower bounds for Flickr8k-full i2t R@10 and Flickr30k-large t2i R@10 are small and occur at high recall, while the R@1/R@5 effects have positive lower bounds across seeds. These results strengthen the claim that the hard-candidate objective generalizes beyond COCO, while still requiring caution on high-recall metrics. For visual inspection, `results/q1_candidate_retrieval_multiseed/candidate_delta_grid.png` plots the mean calibrated deltas over CLIP, and `results/q1_candidate_retrieval_multiseed/candidate_bootstrap_floor.png` plots the worst paired-bootstrap lower bounds across seeds.

### 7.9 Five-caption and Karpathy Retrieval Check

The main benchmark tables above use one caption per image to form a square matrix, which is useful for controlled local comparison but differs from the standard multi-reference retrieval setting. To reduce this external-validity gap, we added a five-caption top-k reranking check and materialized the standard COCO Karpathy split [@karpathy2015deep]. For each image, the evaluator reloads up to five reference captions from the original annotation archive or Karpathy split JSON. Image-to-text retrieval counts any reference caption as a positive target, while text-to-image retrieval maps each caption back to its source image. The first-stage CLIP retriever again supplies top-200 candidates, and `lambda` is selected on validation without test labels. This protocol is implemented in `scripts/run_five_caption_topk_rerank.py`.

| Dataset | Direction | CLIP R@1/R@5/R@10 | Candidate residual R@1/R@5/R@10 | Delta R@1/R@5/R@10 | 95% CI lower bounds |
| --- | --- | --- | --- | --- | --- |
| COCO-1K | i2t | 0.7230 / 0.9080 / 0.9540 | 0.7440 / 0.9220 / 0.9620 | +0.0210 / +0.0140 / +0.0080 | +0.0000 / -0.0010 / -0.0030 |
| COCO-1K | t2i | 0.5106 / 0.7942 / 0.8864 | 0.5638 / 0.8308 / 0.9192 | +0.0532 / +0.0366 / +0.0328 | +0.0436 / +0.0288 / +0.0264 |
| COCO-5K, 3 seeds | i2t | 0.5052 / 0.7489 / 0.8284 | 0.5291+-0.0048 / 0.7664+-0.0031 / 0.8448+-0.0012 | +0.0238 / +0.0175 / +0.0164 | +0.0101 / +0.0067 / +0.0077 |
| COCO-5K, 3 seeds | t2i | 0.3112 / 0.5616 / 0.6745 | 0.3514+-0.0011 / 0.6161+-0.0019 / 0.7227+-0.0010 | +0.0402 / +0.0545 / +0.0482 | +0.0350 / +0.0488 / +0.0436 |
| COCO-Karpathy 5k, 3 seeds | i2t | 0.5090 / 0.7476 / 0.8344 | 0.5299+-0.0010 / 0.7716+-0.0035 / 0.8510+-0.0009 | +0.0209 / +0.0240 / +0.0166 | +0.0104 / +0.0122 / +0.0086 |
| COCO-Karpathy 5k, 3 seeds | t2i | 0.3068 / 0.5647 / 0.6722 | 0.3558+-0.0010 / 0.6189+-0.0008 / 0.7243+-0.0019 | +0.0490 / +0.0542 / +0.0521 | +0.0437 / +0.0495 / +0.0464 |
| Flickr8k-full | i2t | 0.7190 / 0.9170 / 0.9540 | 0.7490 / 0.9380 / 0.9670 | +0.0300 / +0.0210 / +0.0130 | +0.0090 / +0.0070 / +0.0020 |
| Flickr8k-full | t2i | 0.5548 / 0.8122 / 0.8918 | 0.5998 / 0.8542 / 0.9236 | +0.0450 / +0.0420 / +0.0318 | +0.0352 / +0.0344 / +0.0256 |
| Flickr30k-large | i2t | 0.8020 / 0.9590 / 0.9820 | 0.8380 / 0.9630 / 0.9840 | +0.0360 / +0.0040 / +0.0020 | +0.0180 / -0.0050 / -0.0060 |
| Flickr30k-large | t2i | 0.6100 / 0.8428 / 0.9026 | 0.6686 / 0.8844 / 0.9330 | +0.0586 / +0.0416 / +0.0304 | +0.0482 / +0.0346 / +0.0240 |

![Figure 7. Five-caption top-k reranking delta grid. COCO-Karpathy 5k is positive for all six recall metrics across three seeds.](figures/five_caption_delta_grid.png)

![Figure 8. Worst paired-bootstrap lower bounds for five-caption reranking. COCO-Karpathy has positive lower bounds across all reported metrics; the only negative floors appear in high-recall Flickr30k-large image-to-text checks.](figures/five_caption_bootstrap_floor.png)

These five-caption checks are encouraging: the calibrated residual improves all `54/54` per-seed recall point estimates, and `49/54` per-seed bootstrap lower bounds are positive. Both local COCO-5K and COCO-Karpathy have positive worst-case bootstrap lower bounds across three seeds. The weaker cases are high-recall image-to-text metrics where CLIP is already near saturation, mainly on the single-seed COCO-1K and Flickr30k-large checks. We therefore treat the COCO result as a strong standard-split multi-reference validation, while still avoiding an official leaderboard claim because the work has not been submitted to an external evaluation server and Flickr remains a local split protocol.

## 8. Discussion

### 8.1 What the Experiments Support

The experiments support four conclusions. First, FluidMix can be useful as a residual field-dynamic adapter in a domain-specific setting, especially Pokemon-BLIP image-to-text retrieval. Second, residual anchoring and validation calibration are necessary safeguards; without them, the adapter can destroy CLIP ranking. Third, on the standard benchmark protocols, the original pairwise FluidMix training recipe does not beat CLIP, while three-seed CLIP-LoRA and stronger pretrained foundation models provide strong baselines. Fourth, once the residual objective is aligned with CLIP hard retrieval candidates and regularized toward CLIP score geometry, calibrated FluidMix reranking gives positive three-seed mean deltas on COCO-1K, local COCO-5K, COCO-Karpathy, Flickr8k-full, and Flickr30k-large. The five-caption COCO-Karpathy result is the strongest external-validity improvement: it uses 5000 test images, up to five references per image, three seeds, and positive worst-case paired-bootstrap lower bounds for all six recall metrics. This suggests that the main bottleneck is objective design and calibration, not only the field-dynamic architecture.

### 8.2 Why Raw Residual FluidMix Fails on Standard Benchmarks

The likely explanation is an objective mismatch. Binary hard-negative matching and small-batch contrastive training can learn local pairwise separability without preserving the global retrieval geometry that CLIP already has. Because retrieval depends on the full score matrix, a residual correction that improves some pairwise decisions can still damage ranking globally. The validation-selected `lambda=0` result is therefore informative: the validation set detects that the original residual correction is harmful and rejects it. The hard-candidate variant changes the training distribution from arbitrary pair discrimination to the exact region where reranking decisions are made: high-scoring CLIP confusions. This does not make raw residual scores good, but it makes a small calibrated residual useful.

### 8.3 Correct Paper Positioning

The correct positioning is not "FluidMix completely surpasses CLIP." The evidence refutes that claim for the original pairwise-trained residual recipe on COCO-1K, COCO-5K, Flickr8k-full, Flickr30k-large, and scalable COCO top-k reranking. A defensible Q1-style positioning is: FluidMix is a physically inspired residual VLM adapter whose positive effect is objective- and domain-dependent. Its failures reveal important constraints for field-dynamic multimodal fusion, while the hard-candidate retrieval residual objective demonstrates a concrete, reproducible route to standard-benchmark COCO reranking gains. This is more publishable when framed as a rigorous method-and-evidence paper than as an inflated state-of-the-art claim.

### 8.4 Path Toward a Stronger Version

The next technical steps are clear:

1. Add validation-controlled per-query or uncertainty-aware residual gating rather than one global `lambda`.
2. Add BLIP-2 and other modern cross-modal baselines once the corresponding weights are available locally.
3. Replicate external leaderboard-server or community evaluation scripts where the target venue requires them; COCO Karpathy is now reproduced locally, but it is still not an official server submission.
4. Pretrain a CLIP-free FluidMix model at scale before claiming CLIP replacement.
5. Expand FluidMix variants from three to five seeds if the target journal demands stronger statistical stability.

## 9. Threats to Validity

The main internal validity risk is that validation calibration can hide residual failure. We address this by reporting raw residual FluidMix and selected `lambda`, not only the calibrated result. The main external validity risk is that the main COCO/Flickr tables use one caption per image and therefore are not leaderboard-equivalent to standard five-caption retrieval. The five-caption checks reduce this risk substantially, especially the three-seed COCO-Karpathy 5k check, but they do not equal an official external evaluation-server submission. Some high-recall metrics have weaker bootstrap support than the main R@1/R@5 results, especially COCO-1K i2t R@10 and Flickr30k-large i2t R@5/R@10 in the single-seed five-caption checks. The main construct validity risk is that gains on Pokemon-BLIP may come from residual adaptation rather than the physical form of the dynamics. The ablations partially address this, but more controlled comparisons are needed. The main missing baseline is BLIP-2, because BLIP-2 weights were not present in the verified local weight cache.

## 10. Conclusion

FluidMix implements a liquid-inspired advection-diffusion-reaction fusion mechanism for image-text matching. Its model design is well defined, reproducible, and mathematically connected to finite-step field dynamics. The empirical conclusion is now sharper but still bounded. FluidMix improves CLIP on Pokemon-BLIP retrieval under paired-bootstrap testing. The original residual training recipe fails on standard COCO/Flickr benchmarks and scalable COCO top-k reranking, but a hard-candidate retrieval residual objective with CLIP score-geometry regularization obtains validation-calibrated top-k gains across three seeds on COCO-1K, local COCO-5K, Flickr8k-full, and Flickr30k-large. Five-caption checks further show positive point estimates under a multi-reference retrieval definition; the strongest such evidence is COCO-Karpathy 5k, where three seeds improve all six recall metrics with positive bootstrap lower bounds. Calibration safely falls back to CLIP when residual corrections are harmful, while three-seed CLIP-LoRA and modern foundation models remain strong controls. The resulting paper is most credible as a rigorous residual-adapter and evidence-bound method study: it identifies a promising physical prior, demonstrates a domain-specific gain, diagnoses why naive residual training fails, and shows that retrieval-aligned candidate training can turn the same field-dynamic adapter into a useful standard-benchmark reranker without claiming to replace pretrained VLMs.

## Data and Code Availability

The implementation, configurations, processed local datasets, cached features, adapter checkpoints, result tables, and visualizations are organized under `D:/work_space/Liquid_mix`. The main result summary is in `results/q1_benchmark_synthesis`, the multi-seed candidate-reranking summary is in `results/q1_candidate_retrieval_multiseed`, the multi-reference retrieval summary is in `results/q1_five_caption_summary`, the paper figures are in `paper/figures`, the COCO-Karpathy feature cache is in `results/q1_fluidmix_coco_karpathy`, the multi-seed CLIP-LoRA summary is in `results/q1_clip_lora_multiseed`, and the generated DOCX manuscript is `word/FluidMix_Q1_Manuscript.docx`. The DOCX was structurally generated from the Markdown manuscript, but page-render visual QA could not be completed on this machine because LibreOffice/`soffice` is unavailable. Reproducible commands include:

```powershell
py scripts\run_foundation_vlm_benchmark.py --config configs\benchmark_coco1k.yaml --output results\q1_foundation_vlm_coco1k --models clip_b32_openai,openclip_b32_laion2b,siglip_b16_google,blip_itm_coco --batch-size 64 --blip-batch-size 16 --blip-max-items 200 --device auto
py scripts\run_foundation_vlm_benchmark.py --config configs\benchmark_coco5k.yaml --output results\q1_foundation_vlm_coco5k --models clip_b32_openai,openclip_b32_laion2b,siglip_b16_google --batch-size 64 --device auto
curl.exe -L --retry 3 --retry-delay 5 -o data\splits\dataset_coco_karpathy.json "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download="
py scripts\download_dataset.py --config configs\benchmark_coco_karpathy.yaml --force
py scripts\run_foundation_vlm_benchmark.py --config configs\benchmark_coco_karpathy.yaml --output results\q1_foundation_vlm_coco_karpathy --models clip_b32_openai --batch-size 64 --device auto
py scripts\cache_clip_features.py --config configs\benchmark_coco_karpathy.yaml --output results\q1_fluidmix_coco_karpathy --batch-size 64 --device auto
py scripts\build_paper_figures.py
py scripts\build_pandoc_docx.py --input paper\fluidmix_manuscript_q1.md --output word\FluidMix_Q1_Manuscript.docx --paper-copy paper\FluidMix_Q1_Manuscript.docx
py scripts\run_clip_vlm_experiment.py --config configs\benchmark_coco1k.yaml --output results\q1_fluidmix_coco1k
py scripts\run_calibrated_fluidmix_analysis.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --adapter results\q1_fluidmix_coco1k\residual_fluidmix\adapter.pt --output results\q1_fluidmix_coco1k_calibrated
py scripts\run_clip_lora_baseline.py --config configs\benchmark_coco1k.yaml --output results\q1_clip_lora_coco1k_seed7 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 7
py scripts\run_clip_lora_baseline.py --config configs\benchmark_coco1k.yaml --output results\q1_clip_lora_coco1k_seed13 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 13
py scripts\run_clip_lora_baseline.py --config configs\benchmark_coco1k.yaml --output results\q1_clip_lora_coco1k_seed21 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 21
py scripts\run_clip_lora_baseline.py --config configs\benchmark_flickr8k_full.yaml --output results\q1_clip_lora_flickr8k_full_seed7 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 7
py scripts\run_clip_lora_baseline.py --config configs\benchmark_flickr8k_full.yaml --output results\q1_clip_lora_flickr8k_full_seed13 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 13
py scripts\run_clip_lora_baseline.py --config configs\benchmark_flickr8k_full.yaml --output results\q1_clip_lora_flickr8k_full_seed21 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 21
py scripts\run_clip_lora_baseline.py --config configs\benchmark_flickr30k_large.yaml --output results\q1_clip_lora_flickr30k_large_seed7 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 7
py scripts\run_clip_lora_baseline.py --config configs\benchmark_flickr30k_large.yaml --output results\q1_clip_lora_flickr30k_large_seed13 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 13
py scripts\run_clip_lora_baseline.py --config configs\benchmark_flickr30k_large.yaml --output results\q1_clip_lora_flickr30k_large_seed21 --epochs 2 --batch-size 16 --eval-batch-size 64 --lr 1e-5 --amp --device auto --seed 21
py scripts\summarize_clip_lora_seeds.py
py scripts\run_topk_fluidmix_rerank.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --adapter results\q1_fluidmix_coco1k\residual_fluidmix\adapter.pt --output results\q1_topk_rerank_coco1k --top-k 200 --score-batch-size 512 --bootstrap 5000 --device auto
py scripts\run_topk_fluidmix_rerank.py --config configs\benchmark_coco5k.yaml --val-feature-cache results\q1_fluidmix_coco1k --test-feature-file results\q1_foundation_vlm_coco5k\clip_b32_openai_features.pt --adapter results\q1_fluidmix_coco1k\residual_fluidmix\adapter.pt --output results\q1_topk_rerank_coco5k --top-k 200 --score-batch-size 512 --bootstrap 5000 --device auto
py scripts\run_retrieval_first_residual.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --output results\q1_retrieval_first_coco1k --epochs 10 --batch-size 64 --eval-batch-size 512 --lr 1e-4 --geometry-weight 10 --scale-weight 0.1 --scale-init 0.05
py scripts\run_topk_fluidmix_rerank.py --config configs\benchmark_coco5k.yaml --val-feature-cache results\q1_fluidmix_coco1k --test-feature-file results\q1_foundation_vlm_coco5k\clip_b32_openai_features.pt --adapter results\q1_retrieval_first_coco1k\residual_fluidmix\adapter.pt --output results\q1_retrieval_first_coco5k_topk --top-k 200 --lambda-min -2.0 --score-batch-size 512 --bootstrap 5000 --device auto
py scripts\run_candidate_retrieval_residual.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --output results\q1_candidate_retrieval_coco1k --epochs 5 --batch-size 64 --eval-batch-size 512 --hard-candidates 64 --selection-top-k 200 --lr 1e-4 --geometry-weight 10 --scale-weight 0.1 --scale-init 0.05 --seed 13
py scripts\run_candidate_retrieval_residual.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --output results\q1_candidate_retrieval_coco1k_seed7 --epochs 5 --batch-size 64 --eval-batch-size 512 --hard-candidates 64 --selection-top-k 200 --lr 1e-4 --geometry-weight 10 --scale-weight 0.1 --scale-init 0.05 --seed 7
py scripts\run_candidate_retrieval_residual.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --output results\q1_candidate_retrieval_coco1k_seed21 --epochs 5 --batch-size 64 --eval-batch-size 512 --hard-candidates 64 --selection-top-k 200 --lr 1e-4 --geometry-weight 10 --scale-weight 0.1 --scale-init 0.05 --seed 21
py scripts\run_topk_fluidmix_rerank.py --config configs\benchmark_coco5k.yaml --val-feature-cache results\q1_fluidmix_coco1k --test-feature-file results\q1_foundation_vlm_coco5k\clip_b32_openai_features.pt --adapter results\q1_candidate_retrieval_coco1k\residual_fluidmix\adapter.pt --output results\q1_candidate_retrieval_coco5k_topk --top-k 200 --lambda-min -2.0 --score-batch-size 512 --bootstrap 5000 --device auto
py scripts\run_candidate_retrieval_residual.py --config configs\benchmark_flickr8k_full.yaml --feature-cache results\q1_fluidmix_flickr8k_full --output results\q1_candidate_retrieval_flickr8k_full --epochs 5 --batch-size 64 --eval-batch-size 512 --hard-candidates 64 --selection-top-k 200 --lr 1e-4 --geometry-weight 10 --scale-weight 0.1 --scale-init 0.05 --seed 13
py scripts\run_topk_fluidmix_rerank.py --config configs\benchmark_flickr8k_full.yaml --feature-cache results\q1_fluidmix_flickr8k_full --adapter results\q1_candidate_retrieval_flickr8k_full\residual_fluidmix\adapter.pt --output results\q1_candidate_retrieval_flickr8k_full_topk --top-k 200 --lambda-min -2.0 --score-batch-size 512 --bootstrap 5000 --device auto
py scripts\run_candidate_retrieval_residual.py --config configs\benchmark_flickr30k_large.yaml --feature-cache results\q1_fluidmix_flickr30k_large --output results\q1_candidate_retrieval_flickr30k_large --epochs 5 --batch-size 64 --eval-batch-size 512 --hard-candidates 64 --selection-top-k 200 --lr 1e-4 --geometry-weight 10 --scale-weight 0.1 --scale-init 0.05 --seed 13
py scripts\run_topk_fluidmix_rerank.py --config configs\benchmark_flickr30k_large.yaml --feature-cache results\q1_fluidmix_flickr30k_large --adapter results\q1_candidate_retrieval_flickr30k_large\residual_fluidmix\adapter.pt --output results\q1_candidate_retrieval_flickr30k_large_topk --top-k 200 --lambda-min -2.0 --score-batch-size 512 --bootstrap 5000 --device auto
py scripts\summarize_candidate_retrieval_seeds.py
py scripts\plot_candidate_retrieval_summary.py --input-dir results\q1_candidate_retrieval_multiseed
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco1k.yaml --feature-cache results\q1_fluidmix_coco1k --adapter results\q1_candidate_retrieval_coco1k\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco1k --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco5k.yaml --feature-cache results\q1_fluidmix_coco1k --val-feature-cache results\q1_fluidmix_coco1k --test-feature-file results\q1_foundation_vlm_coco5k\clip_b32_openai_features.pt --adapter results\q1_candidate_retrieval_coco1k\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco5k --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco5k.yaml --feature-cache results\q1_fluidmix_coco1k --val-feature-cache results\q1_fluidmix_coco1k --test-feature-file results\q1_foundation_vlm_coco5k\clip_b32_openai_features.pt --adapter results\q1_candidate_retrieval_coco1k_seed7\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco5k_seed7 --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco5k.yaml --feature-cache results\q1_fluidmix_coco1k --val-feature-cache results\q1_fluidmix_coco1k --test-feature-file results\q1_foundation_vlm_coco5k\clip_b32_openai_features.pt --adapter results\q1_candidate_retrieval_coco1k_seed21\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco5k_seed21 --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco_karpathy.yaml --feature-cache results\q1_fluidmix_coco_karpathy --adapter results\q1_candidate_retrieval_coco1k\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco_karpathy --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco_karpathy.yaml --feature-cache results\q1_fluidmix_coco_karpathy --adapter results\q1_candidate_retrieval_coco1k_seed7\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco_karpathy_seed7 --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_coco_karpathy.yaml --feature-cache results\q1_fluidmix_coco_karpathy --adapter results\q1_candidate_retrieval_coco1k_seed21\residual_fluidmix\adapter.pt --output results\q1_five_caption_coco_karpathy_seed21 --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_flickr8k_full.yaml --feature-cache results\q1_fluidmix_flickr8k_full --adapter results\q1_candidate_retrieval_flickr8k_full\residual_fluidmix\adapter.pt --output results\q1_five_caption_flickr8k_full --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\run_five_caption_topk_rerank.py --config configs\benchmark_flickr30k_large.yaml --feature-cache results\q1_fluidmix_flickr30k_large --adapter results\q1_candidate_retrieval_flickr30k_large\residual_fluidmix\adapter.pt --output results\q1_five_caption_flickr30k_large --top-k 200 --max-captions 5 --bootstrap 5000 --device auto
py scripts\summarize_five_caption_results.py
py -m pytest tests -q
```

## References

References are provided in `references.bib`.
