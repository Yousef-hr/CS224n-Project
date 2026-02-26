"""
Train JEPA baseline text classifier (frozen encoder + MLP predictor).

Thin wrapper: composable CLI + model-specific args, calls shared runner.
"""

from pathlib import Path

import torch

from utils.metrics import covariance_spectrum, effective_rank, variance_ratio

from ...train import run_text_classification_train, __encode_batch__
from ...cli import make_train_parser, build_data_spec, build_train_spec

from .model import BaselineJEPATextClassifier


@torch.no_grad()
def __get_representation_metrics__(model, loader, ctx) -> dict[str, float]:
    model.eval_mode()
    if ctx.label_embeddings is None or not ctx.extra or "encoder" not in ctx.extra:
        return {}
        
    encoder = ctx.extra["encoder"]
    all_pred = []
    all_tgt = []

    for inputs, labels in loader:
        labels = labels.to(ctx.device)
        input_emb = __encode_batch__(encoder, inputs, ctx.device, ctx.use_amp)
        pred_emb = model.forward(input_emb, ctx)
        pred_emb = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-12)
        tgt = ctx.label_embeddings[labels]
        all_pred.append(pred_emb.detach().cpu())
        all_tgt.append(tgt.detach().cpu())

    pred_all = torch.cat(all_pred, dim=0)
    tgt_all = torch.cat(all_tgt, dim=0)

    return {
        "pred_effective_rank": effective_rank(pred_all),
        "target_effective_rank": effective_rank(tgt_all),
        "variance_ratio": variance_ratio(pred_all, tgt_all),
        "pred_cov_top1_eig": float(covariance_spectrum(pred_all)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(tgt_all)[0].item()),
    }


def main() -> None:
    parser = make_train_parser()
    parser.add_argument("--predictor_hidden", type=int, default=1024)
    parser.add_argument("--baseline_dropout", type=float, default=0.1)
    args = parser.parse_args()

    data = build_data_spec(args)
    train = build_train_spec(args, metrics_csv_default=Path(args.save_dir) / "training_metrics_jepa_baseline.csv")

    def _factory(labels_list: list[str], embed_dim: int):
        return BaselineJEPATextClassifier(
            embed_dim=embed_dim,
            predictor_hidden_dim=args.predictor_hidden,
            baseline_dropout=args.baseline_dropout,
        )

    run_text_classification_train(
        model_factory=_factory,
        data=data,
        train=train,
        extra_eval_metrics=__get_representation_metrics__,
        save_name="best_jepa_baseline.pt",
    )


if __name__ == "__main__":
    main()
