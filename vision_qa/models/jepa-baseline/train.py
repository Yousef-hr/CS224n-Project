"""Train JEPA Baseline VQA (frozen CLIP + MLP predictor).

Supports ``--embedding_cache_dir`` to precompute and reuse frozen CLIP
embeddings, skipping the encoder at train time for much faster epochs.
"""

import argparse
from pathlib import Path

import torch

from utils.metrics import covariance_spectrum, effective_rank, variance_ratio

from vision_qa.base import DataSpec, TrainSpec
from vision_qa.train import prepare_and_run_cached, run_vision_qa_train

from .model import VisionQAJEPABaseline


# ---- extra eval metrics (live CLIP encoding) ----

@torch.no_grad()
def _repr_metrics(model, loader, ctx) -> dict[str, float]:
    model.eval_mode()
    all_pred, all_tgt = [], []

    for batch in loader:
        batch.answer_indices = batch.answer_indices.to(ctx.device)
        batch.images = batch.images.to(ctx.device)
        embeddings = model.encode_inputs(batch, ctx)
        outputs = model.forward(embeddings, ctx)

        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        targets = model._choice_embs[
            torch.arange(len(batch.answer_indices), device=ctx.device),
            batch.answer_indices,
        ]
        all_pred.append(pred_norm.cpu())
        all_tgt.append(targets.cpu())

    pred_all = torch.cat(all_pred)
    tgt_all = torch.cat(all_tgt)
    return {
        "pred_effective_rank": effective_rank(pred_all),
        "target_effective_rank": effective_rank(tgt_all),
        "variance_ratio": variance_ratio(pred_all, tgt_all),
        "pred_cov_top1_eig": float(covariance_spectrum(pred_all)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(tgt_all)[0].item()),
    }


# ---- extra eval metrics (cached embeddings) ----

@torch.no_grad()
def _repr_metrics_cached(model, loader, ctx) -> dict[str, float]:
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
        targets = choice_embs[torch.arange(len(answer_indices), device=ctx.device), answer_indices]
        all_pred.append(pred_norm.cpu())
        all_tgt.append(targets.cpu())

    pred_all = torch.cat(all_pred)
    tgt_all = torch.cat(all_tgt)
    return {
        "pred_effective_rank": effective_rank(pred_all),
        "target_effective_rank": effective_rank(tgt_all),
        "variance_ratio": variance_ratio(pred_all, tgt_all),
        "pred_cov_top1_eig": float(covariance_spectrum(pred_all)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(tgt_all)[0].item()),
    }


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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--metrics_csv", type=str, default=None)
    parser.add_argument("--use_image", action="store_true", default=True)
    parser.add_argument("--no_use_image", action="store_false", dest="use_image")
    parser.add_argument("--embedding_cache_dir", type=str, default=None,
                        help="If set, precompute CLIP embeddings once and train from cache")
    parser.add_argument("--precompute_batch_size", type=int, default=64)
    args = parser.parse_args()

    model = VisionQAJEPABaseline(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
    )

    train_spec = TrainSpec(
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        metrics_csv=args.metrics_csv or str(Path(args.save_dir) / "training_metrics_jepa_baseline.csv"),
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
            extra_eval_metrics=_repr_metrics_cached,
            save_name="best_jepa_baseline.pt",
        )
    else:
        data = DataSpec(
            dataset=args.dataset, subset=args.subset, cache_dir=args.cache_dir,
            batch_size=args.batch_size, num_workers=0, use_image=args.use_image,
        )
        run_vision_qa_train(
            model=model, data=data, train=train_spec,
            image_transform=model.get_image_transform(),
            extra_eval_metrics=_repr_metrics,
            save_name="best_jepa_baseline.pt",
        )


if __name__ == "__main__":
    main()
