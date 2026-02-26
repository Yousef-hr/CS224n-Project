"""
Train SOTA baseline text classifier (frozen encoder + MLP → logits).

Thin wrapper: composable CLI + model-specific args, calls shared runner.
"""

from pathlib import Path
import torch

from utils.metrics import covariance_spectrum, effective_rank

from ...train import run_text_classification_train, __encode_batch__
from ...cli import make_train_parser, build_data_spec, build_train_spec

from .model import SOTABaselineTextClassifier


def add_sota_baseline_args(parser):
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024], help="MLP hidden dims, e.g. 1024")
    parser.add_argument("--dropout", type=float, default=0.1)


def __extra_eval_logit_metrics__(model, loader, ctx) -> dict[str, float]:
    model.eval()
    encoder = ctx.extra.get("encoder") if ctx.extra else None
    if encoder is None:
        return {}

    all_logits = []
    for inputs, _labels in loader:
        input_emb = __encode_batch__(encoder, inputs, ctx.device, ctx.use_amp)
        logits = model.forward(input_emb, ctx)
        all_logits.append(logits.detach().cpu())
    logits_all = torch.cat(all_logits, dim=0)
    spectrum = covariance_spectrum(logits_all)

    return {
        "logit_effective_rank": effective_rank(logits_all),
        "logit_cov_top1_eig": float(spectrum[0].item()) if spectrum.numel() else 0.0,
    }


def main() -> None:
    parser = make_train_parser()
    add_sota_baseline_args(parser)
    args = parser.parse_args()

    data = build_data_spec(args)
    train = build_train_spec(args, metrics_csv_default=Path(args.save_dir) / "training_metrics_sota.csv")

    def __model_factory__(labels_list: list[str], embed_dim: int):
        return SOTABaselineTextClassifier(
            embed_dim=embed_dim,
            num_classes=len(labels_list),
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )

    run_text_classification_train(
        model_factory=__model_factory__,
        data=data,
        train=train,
        extra_eval_metrics=__extra_eval_logit_metrics__,
        save_name="best_sota_baseline.pt",
    )


if __name__ == "__main__":
    main()
