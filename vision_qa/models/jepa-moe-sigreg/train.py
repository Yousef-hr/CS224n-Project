"""Train JEPA MoE + SigReg VQA (frozen CLIP + projection + MoE + BCS regulariser).

Supports ``--embedding_cache_dir`` for precomputed CLIP embeddings.
"""

import argparse
from pathlib import Path

import torch

from utils.metrics import (
    conditional_routing_entropy,
    covariance_spectrum,
    effective_rank,
    expert_usage_entropy,
    variance_ratio,
)

from vision_qa.base import DataSpec, TrainSpec
from vision_qa.train import prepare_and_run_cached, run_vision_qa_train

from .model import VisionQAJEPAMoESigReg


def _moe_repr(pred_all, tgt_all, gate_all):
    usage_h, usage_h_norm = expert_usage_entropy(gate_all)
    cond_h, cond_h_norm = conditional_routing_entropy(gate_all)
    return {
        "pred_effective_rank": effective_rank(pred_all),
        "target_effective_rank": effective_rank(tgt_all),
        "variance_ratio": variance_ratio(pred_all, tgt_all),
        "pred_cov_top1_eig": float(covariance_spectrum(pred_all)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(tgt_all)[0].item()),
        "expert_usage_entropy": usage_h,
        "expert_usage_entropy_norm": usage_h_norm,
        "conditional_routing_entropy": cond_h,
        "conditional_routing_entropy_norm": cond_h_norm,
    }


@torch.no_grad()
def _moe_sigreg_metrics(model, loader, ctx) -> dict[str, float]:
    model.eval_mode()
    all_pred, all_tgt, all_gate = [], [], []
    for batch in loader:
        batch.answer_indices = batch.answer_indices.to(ctx.device)
        batch.images = batch.images.to(ctx.device)
        embeddings = model.encode_inputs(batch, ctx)
        pred_emb, gate_probs, _ = model.forward_with_diagnostics(embeddings, ctx)
        pred_norm = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-12)
        targets = model._choice_embs[
            torch.arange(len(batch.answer_indices), device=ctx.device),
            batch.answer_indices,
        ]
        all_pred.append(pred_norm.cpu())
        all_tgt.append(targets.cpu())
        all_gate.append(gate_probs.cpu())
    return _moe_repr(torch.cat(all_pred), torch.cat(all_tgt), torch.cat(all_gate))


@torch.no_grad()
def _moe_sigreg_metrics_cached(model, loader, ctx) -> dict[str, float]:
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
        targets = choice_embs[torch.arange(len(answer_indices), device=ctx.device), answer_indices]
        all_pred.append(pred_norm.cpu())
        all_tgt.append(targets.cpu())
        all_gate.append(gate_probs.cpu())
    return _moe_repr(torch.cat(all_pred), torch.cat(all_tgt), torch.cat(all_gate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="science_qa", choices=["science_qa"])
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--sigreg_weight", type=float, default=0.10)
    parser.add_argument("--sigreg_num_slices", type=int, default=256)
    parser.add_argument("--sigreg_lmbd", type=float, default=10.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--metrics_csv", type=str, default=None)
    parser.add_argument("--use_image", action="store_true", default=True)
    parser.add_argument("--no_use_image", action="store_false", dest="use_image")
    parser.add_argument("--embedding_cache_dir", type=str, default=None)
    parser.add_argument("--precompute_batch_size", type=int, default=64)
    args = parser.parse_args()

    model = VisionQAJEPAMoESigReg(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        hidden_dim=args.hidden_dim,
        moe_num_experts=args.moe_num_experts,
        lr=args.lr,
        num_slices=args.sigreg_num_slices,
        lmbd=args.sigreg_lmbd,
        sigreg_weight=args.sigreg_weight,
    )

    train_spec = TrainSpec(
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        metrics_csv=args.metrics_csv or str(Path(args.save_dir) / "training_metrics_jepa_moe_sigreg.csv"),
    )

    if args.embedding_cache_dir:
        prepare_and_run_cached(
            model=model,
            dataset_name=args.dataset,
            subset=args.subset,
            hf_cache_dir=args.cache_dir,
            use_image=args.use_image,
            embedding_cache_dir=args.embedding_cache_dir,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            precompute_batch_size=args.precompute_batch_size,
            batch_size=args.batch_size,
            train_spec=train_spec,
            extra_eval_metrics=_moe_sigreg_metrics_cached,
            save_name="best_jepa_moe_sigreg.pt",
        )
    else:
        data = DataSpec(
            dataset=args.dataset, subset=args.subset, cache_dir=args.cache_dir,
            batch_size=args.batch_size, num_workers=0, use_image=args.use_image,
        )
        run_vision_qa_train(
            model=model, data=data, train=train_spec,
            image_transform=model.get_image_transform(),
            extra_eval_metrics=_moe_sigreg_metrics,
            save_name="best_jepa_moe_sigreg.pt",
        )


if __name__ == "__main__":
    main()
