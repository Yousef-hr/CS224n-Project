"""
Test/evaluate JEPA baseline text classifier.

Thin wrapper: composable CLI + model-specific args, calls shared evaluator.
"""

from ...eval import run_text_classification_eval
from ...cli import make_eval_parser, build_data_spec, build_eval_spec

from .model import BaselineJEPATextClassifier
from .train import add_jepa_baseline_args


def main() -> None:
    parser = make_eval_parser()
    parser.set_defaults(checkpoint="checkpoints/best_jepa_baseline.pt")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus", help="Alias for --subset when dataset=clinc_oos")
    add_jepa_baseline_args(parser)
    args = parser.parse_args()
    if args.dataset == "clinc_oos" and args.subset is None:
        args.subset = args.clinc_config

    data = build_data_spec(args)
    eval_spec = build_eval_spec(args)

    def __model_factory__(labels_list: list[str], embed_dim: int):
        return BaselineJEPATextClassifier(
            embed_dim=embed_dim,
            predictor_hidden_dim=args.predictor_hidden,
            baseline_dropout=args.baseline_dropout,
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
