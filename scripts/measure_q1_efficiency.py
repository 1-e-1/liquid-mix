from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch
from torch import nn
import yaml

from liquid_mix.models import FluidMixFusion
from liquid_mix.rerankers import trainable_parameter_count
from run_clip_vlm_experiment import resolve_device
from run_extended_q1_experiments import load_dataset_features, make_model, resolve


def estimate_supported_flops(model: nn.Module, image: torch.Tensor, text: torch.Tensor) -> int:
    total = 0
    handles = []

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal total
        instances = inputs[0].numel() // module.in_features
        total += int(instances * 2 * module.in_features * module.out_features)

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal total
        kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        total += int(output.numel() * 2 * kernel)

    def attention_hook(module: nn.MultiheadAttention, inputs, output):
        nonlocal total
        query = inputs[0]
        key = inputs[1]
        batch, query_tokens, dim = query.shape
        key_tokens = key.shape[1]
        total += int(batch * (8 * (query_tokens + key_tokens) * dim * dim + 4 * query_tokens * key_tokens * dim))

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(attention_hook))
    with torch.no_grad():
        model(image, text)
    for handle in handles:
        handle.remove()

    for module in model.modules():
        if isinstance(module, FluidMixFusion):
            batch = image.shape[0]
            cells = batch * module.laplacian_kernel.shape[0] * module.field_grid**2
            per_step = 0
            if module.use_advection:
                per_step += 8 * cells
            if module.use_diffusion:
                per_step += cells
            if module.use_reaction:
                per_step += 2 * cells
            per_step += 4 * cells
            total += int(module.steps * per_step)
    return int(total)


def timed_forward(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(int(warmup)):
            model(image, text)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = []
        for _ in range(int(repetitions)):
            start = time.perf_counter()
            model(image, text)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(elapsed)), float(sorted(elapsed)[int(0.95 * (len(elapsed) - 1))])


def peak_memory_mb(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    device: torch.device,
    backward: bool,
) -> float:
    if device.type != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if backward:
        model.train()
        model.zero_grad(set_to_none=True)
        model(image, text).mean().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(image, text)
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extended_q1_experiments.yaml")
    parser.add_argument("--results", default="results/q1_extended")
    parser.add_argument("--dataset", default="coco_karpathy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()

    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    protocol = config["protocol"]
    output_dir = resolve(args.results)
    dataset_spec = next(spec for spec in config["datasets"] if spec["name"] == args.dataset)
    device = resolve_device(args.device)
    features, _, _ = load_dataset_features(dataset_spec, device)
    top_k = int(dataset_spec.get("top_k", protocol["top_k"]))
    image_single = features["test"]["image"][:1].repeat(top_k, 1).to(device)
    text_single = features["test"]["text"][:top_k].to(device)
    image_train = features["train"]["image"][:512].to(device)
    text_train = features["train"]["text"][:512].to(device)
    model_names = list(
        dict.fromkeys(
            config["models"] + list(config["ablations"]) + ["parameter_matched_mlp"]
        )
    )
    rows = []
    for model_name in model_names:
        run_dir = output_dir / "runs" / args.dataset / model_name
        checkpoint_path = run_dir / "checkpoint.pt"
        completed_path = run_dir / "completed.json"
        if not checkpoint_path.exists() or not completed_path.exists():
            continue
        model, _ = make_model(
            model_name,
            protocol,
            int(features["train"]["image"].shape[1]),
            config["ablations"],
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        cpu_model, _ = make_model(
            model_name,
            protocol,
            int(features["train"]["image"].shape[1]),
            config["ablations"],
        )
        cpu_model.load_state_dict(checkpoint["model"])
        cpu_model.eval()
        flops = estimate_supported_flops(
            cpu_model,
            features["test"]["image"][:1],
            features["test"]["text"][:1],
        )
        median_ms, p95_ms = timed_forward(
            model,
            image_single,
            text_single,
            device,
            warmup=10,
            repetitions=int(args.repetitions),
        )
        inference_memory = peak_memory_mb(model, image_single, text_single, device, backward=False)
        training_memory = peak_memory_mb(model, image_train, text_train, device, backward=True)
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": args.dataset,
                "model": model_name,
                "trainable_parameters": trainable_parameter_count(model),
                "estimated_flops_per_pair": flops,
                "peak_inference_vram_mb_topk": inference_memory,
                "peak_training_vram_mb_512_pairs": training_memory,
                "single_query_topk_latency_median_ms": median_ms,
                "single_query_topk_latency_p95_ms": p95_ms,
                "top_k": top_k,
                "training_seconds": completed["training_seconds"],
                "device": str(device),
                "flops_scope": "linear, convolution, attention, and explicit ADR finite-difference operations",
            }
        )
        del model, cpu_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "efficiency.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
