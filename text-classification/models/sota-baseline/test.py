"""
Test/evaluate SOTA baseline text classifier (frozen encoder + MLP → logits).

Thin wrapper: composable CLI + model-specific args, calls shared evaluator.
"""

from ...eval import run_text_classification_eval
from ...cli import make_eval_parser, build_data_spec, build_eval_spec

from .model import SOTABaselineTextClassifier
from .train import add_sota_baseline_args

def main() -> None:
    parser = make_eval_parser()
    parser.set_defaults(checkpoint="checkpoints/best_sota_baseline.pt")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus", help="Alias for --subset when dataset=clinc_oos")
    add_sota_baseline_args(parser)
    args = parser.parse_args()
    if args.dataset == "clinc_oos" and args.subset is None:
        args.subset = args.clinc_config

    data = build_data_spec(args)
    eval_spec = build_eval_spec(args)

    def __model_factory__(labels_list: list[str], embed_dim: int):
        return SOTABaselineTextClassifier(
            embed_dim=embed_dim,
            num_classes=len(labels_list),
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )

    run_text_classification_eval(
        model_factory=__model_factory__,
        data=data,
        eval=eval_spec,
        checkpoint=args.checkpoint,
        strict=not getattr(args, "no_strict", False),
    )


if __name__ == "__main__":
    main()
