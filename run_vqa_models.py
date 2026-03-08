"""Train all 5 VQA model variants on ScienceQA sequentially.

The four JEPA models share a single precomputed CLIP embedding cache
(built once by the first model, reused by the rest).  The SOTA cross-
entropy baseline runs with live CLIP encoding because its text encoding
strategy (question + all choices concatenated) is structurally different.

Per-model training metrics are saved to individual CSVs, then merged
into one combined CSV at the end.

Usage
-----
    python run_vqa_models.py --epochs 20 --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------
# Dynamic import helper (handles hyphenated directory names)
# ------------------------------------------------------------------

def _load_class(rel_path: str, class_name: str):
    filepath = _ROOT / rel_path
    mod_name = f"_dyn_{class_name}"
    spec = importlib.util.spec_from_file_location(mod_name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


# ------------------------------------------------------------------
# Extra-eval metric helpers (cached JEPA variants)
# ------------------------------------------------------------------

@torch.no_grad()
def _repr_metrics_cached(model, loader, ctx) -> dict[str, float]:
    from utils.metrics import covariance_spectrum, effective_rank, variance_ratio

    model.eval_mode()
    all_pred, all_tgt = [], []
    for fused_emb, choice_embs, answer_indices, num_choices in loader:
        fused_emb = fused_emb.to(ctx.device)
        choice_embs = choice_embs.to(ctx.device)
        answer_indices = answer_indices.to(ctx.device)
        model._choice_embs = choice_embs
        model._num_choices = num_choices.to(ctx.device)

        outputs = model.forward(fused_emb, ctx)
        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        targets = choice_embs[
            torch.arange(len(answer_indices), device=ctx.device), answer_indices
        ]
        all_pred.append(pred_norm.cpu())
        all_tgt.append(targets.cpu())

    p, t = torch.cat(all_pred), torch.cat(all_tgt)
    return {
        "pred_effective_rank": effective_rank(p),
        "target_effective_rank": effective_rank(t),
        "variance_ratio": variance_ratio(p, t),
        "pred_cov_top1_eig": float(covariance_spectrum(p)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(t)[0].item()),
    }


@torch.no_grad()
def _moe_metrics_cached(model, loader, ctx) -> dict[str, float]:
    from utils.metrics import (
        conditional_routing_entropy,
        covariance_spectrum,
        effective_rank,
        expert_usage_entropy,
        variance_ratio,
    )

    model.eval_mode()
    all_pred, all_tgt, all_gate = [], [], []
    for fused_emb, choice_embs, answer_indices, num_choices in loader:
        fused_emb = fused_emb.to(ctx.device)
        choice_embs = choice_embs.to(ctx.device)
        answer_indices = answer_indices.to(ctx.device)
        model._choice_embs = choice_embs
        model._num_choices = num_choices.to(ctx.device)

        pred_emb, gate_probs, _ = model.forward_with_diagnostics(fused_emb, ctx)
        pred_norm = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-12)
        targets = choice_embs[
            torch.arange(len(answer_indices), device=ctx.device), answer_indices
        ]
        all_pred.append(pred_norm.cpu())
        all_tgt.append(targets.cpu())
        all_gate.append(gate_probs.cpu())

    p, t, g = torch.cat(all_pred), torch.cat(all_tgt), torch.cat(all_gate)
    usage_h, usage_h_norm = expert_usage_entropy(g)
    cond_h, cond_h_norm = conditional_routing_entropy(g)
    return {
        "pred_effective_rank": effective_rank(p),
        "target_effective_rank": effective_rank(t),
        "variance_ratio": variance_ratio(p, t),
        "pred_cov_top1_eig": float(covariance_spectrum(p)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(t)[0].item()),
        "expert_usage_entropy": usage_h,
        "expert_usage_entropy_norm": usage_h_norm,
        "conditional_routing_entropy": cond_h,
        "conditional_routing_entropy_norm": cond_h_norm,
    }


# ------------------------------------------------------------------
# Model definitions (order = training order)
# ------------------------------------------------------------------

MODEL_DEFS: list[dict] = [
    {
        "name": "VQA-JEPA-Baseline",
        "file": "vision_qa/models/jepa-baseline/model.py",
        "class": "VisionQAJEPABaseline",
        "type": "jepa",
        "extra_kwargs": lambda a: {
            "hidden_dim": a.hidden_dim,
            "dropout": a.dropout,
            "lr": a.lr,
        },
        "metrics_fn": _repr_metrics_cached,
        "save_name": "best_jepa_baseline.pt",
        "csv_stem": "training_metrics_jepa_baseline",
    },
    {
        "name": "VQA-JEPA-SigReg",
        "file": "vision_qa/models/jepa-sigreg/model.py",
        "class": "VisionQAJEPASigReg",
        "type": "jepa",
        "extra_kwargs": lambda a: {
            "hidden_dim": a.hidden_dim,
            "dropout": a.dropout,
            "lr": a.lr,
            "sigreg_weight": a.sigreg_weight,
        },
        "metrics_fn": _repr_metrics_cached,
        "save_name": "best_jepa_sigreg.pt",
        "csv_stem": "training_metrics_jepa_sigreg",
    },
    {
        "name": "VQA-JEPA-MoE",
        "file": "vision_qa/models/jepa-moe/model.py",
        "class": "VisionQAJEPAMoE",
        "type": "jepa",
        "extra_kwargs": lambda a: {
            "hidden_dim": a.hidden_dim,
            "moe_num_experts": a.moe_num_experts,
            "lr": a.lr,
        },
        "metrics_fn": _moe_metrics_cached,
        "save_name": "best_jepa_moe.pt",
        "csv_stem": "training_metrics_jepa_moe",
    },
    {
        "name": "VQA-JEPA-MoE-SigReg",
        "file": "vision_qa/models/jepa-moe-sigreg/model.py",
        "class": "VisionQAJEPAMoESigReg",
        "type": "jepa",
        "extra_kwargs": lambda a: {
            "hidden_dim": a.hidden_dim,
            "moe_num_experts": a.moe_num_experts,
            "lr": a.lr,
            "sigreg_weight": a.sigreg_weight,
        },
        "metrics_fn": _moe_metrics_cached,
        "save_name": "best_jepa_moe_sigreg.pt",
        "csv_stem": "training_metrics_jepa_moe_sigreg",
    },
    {
        "name": "VQA-SOTA-Baseline",
        "file": "vision_qa/models/sota-baseline/model.py",
        "class": "VisionQABaseline",
        "type": "sota",
        "extra_kwargs": lambda a: {
            "hidden_dim": a.hidden_dim,
            "dropout": a.dropout,
            "lr": a.lr,
        },
        "metrics_fn": None,
        "save_name": "best_sota_baseline.pt",
        "csv_stem": "training_metrics_sota_baseline",
    },
]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all 5 VQA models on ScienceQA.")

    parser.add_argument("--dataset", default="science_qa")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace dataset cache dir")
    parser.add_argument("--use_image", action="store_true", default=True)
    parser.add_argument("--no_use_image", action="store_false", dest="use_image")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--sigreg_weight", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--embedding_cache_dir", default="precomputed_embeddings",
                        help="Dir to store precomputed CLIP embeddings (shared by all JEPA models)")
    parser.add_argument("--precompute_batch_size", type=int, default=64)
    parser.add_argument("--save_root", default="runs")

    args = parser.parse_args()

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    from vision_qa.base import DataSpec, TrainSpec
    from vision_qa.train import prepare_and_run_cached, run_vision_qa_train

    timings: list[tuple[str, float]] = []
    csv_paths: list[Path] = []

    for mdef in MODEL_DEFS:
        banner = f" {mdef['name']} "
        print(f"\n{'=' * 60}")
        print(f"{banner:=^60}")
        print(f"{'=' * 60}\n")

        t0 = time.time()

        ModelClass = _load_class(mdef["file"], mdef["class"])
        kwargs = mdef["extra_kwargs"](args)
        model = ModelClass(
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            **kwargs,
        )

        run_dir = save_root / mdef["name"] / f"seed_{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = run_dir / f"{mdef['csv_stem']}.csv"
        csv_paths.append(metrics_csv)

        train_spec = TrainSpec(
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
            save_dir=str(run_dir),
            metrics_csv=str(metrics_csv),
        )

        if mdef["type"] == "jepa":
            prepare_and_run_cached(
                model=model,
                dataset_name=args.dataset,
                subset=None,
                hf_cache_dir=args.cache_dir,
                use_image=args.use_image,
                embedding_cache_dir=args.embedding_cache_dir,
                clip_model_name=args.clip_model,
                clip_pretrained=args.clip_pretrained,
                precompute_batch_size=args.precompute_batch_size,
                batch_size=args.batch_size,
                train_spec=train_spec,
                extra_eval_metrics=mdef["metrics_fn"],
                save_name=mdef["save_name"],
            )
        else:
            data = DataSpec(
                dataset=args.dataset,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                use_image=args.use_image,
            )
            run_vision_qa_train(
                model=model,
                data=data,
                train=train_spec,
                image_transform=model.get_image_transform(),
                save_name=mdef["save_name"],
            )

        elapsed = time.time() - t0
        timings.append((mdef["name"], elapsed))
        print(f"\n{mdef['name']} finished in {elapsed / 60:.1f} min")

        del model
        _cleanup()

    # ------------------------------------------------------------------
    # Combine per-model CSVs into one
    # ------------------------------------------------------------------
    combined_csv = save_root / "vqa_all_results.csv"
    all_rows: list[dict[str, str]] = []

    for mdef, csv_path in zip(MODEL_DEFS, csv_paths):
        if not csv_path.exists():
            print(f"Warning: CSV not found for {mdef['name']}: {csv_path}")
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["model"] = mdef["name"]
                all_rows.append(row)

    if all_rows:
        fieldnames = ["model"] + sorted(k for k in all_rows[0] if k != "model")
        with combined_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCombined metrics saved to: {combined_csv}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Training summary")
    print(f"{'=' * 60}")
    for name, elapsed in timings:
        print(f"  {name:30s} {elapsed / 60:6.1f} min")
    total = sum(t for _, t in timings)
    print(f"  {'TOTAL':30s} {total / 60:6.1f} min")


if __name__ == "__main__":
    main()
