"""
Test/evaluate JEPA MoE text classifier.

Thin wrapper: composable CLI + model-specific args, calls shared evaluator.
"""

import torch

from utils.metrics import conditional_routing_entropy, expert_pairwise_cka, expert_usage_entropy

from ...eval import run_text_classification_eval
from ...train import __encode_batch__
from ...cli import make_eval_parser, build_data_spec, build_eval_spec

from .model import MoEJEPATextClassifier
from .train import add_jepa_moe_args


@torch.no_grad()
def __extra_moe_eval_metrics__(model, loader, ctx) -> dict[str, float]:
    if not ctx.extra or "encoder" not in ctx.extra:
        return {}
    encoder = ctx.extra["encoder"]
    all_gate = []
    all_expert_outputs = []
    for inputs, _labels in loader:
        input_emb = __encode_batch__(encoder, inputs, ctx.device, ctx.use_amp)
        _pred, gate_probs, expert_outputs = model.forward_with_diagnostics(input_emb, ctx)
        all_gate.append(gate_probs.detach().cpu())
        all_expert_outputs.append(expert_outputs.detach().cpu())

    gate_probs_all = torch.cat(all_gate, dim=0)
    usage_h, usage_h_norm = expert_usage_entropy(gate_probs_all)
    cond_h, cond_h_norm = conditional_routing_entropy(gate_probs_all)

    expert_outputs_all = torch.cat(all_expert_outputs, dim=0)
    cka_mean, cka_max = expert_pairwise_cka(expert_outputs_all)
    return {
        "expert_usage_entropy": usage_h,
        "expert_usage_entropy_norm": usage_h_norm,
        "conditional_routing_entropy": cond_h,
        "conditional_routing_entropy_norm": cond_h_norm,
        "expert_pairwise_cka_mean": cka_mean,
        "expert_pairwise_cka_max": cka_max,
    }


def main() -> None:
    parser = make_eval_parser()
    parser.set_defaults(checkpoint="checkpoints/best_jepa_moe.pt")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus", help="Alias for --subset when dataset=clinc_oos")
    add_jepa_moe_args(parser)
    args = parser.parse_args()
    if args.dataset == "clinc_oos" and args.subset is None:
        args.subset = args.clinc_config

    data = build_data_spec(args)
    eval_spec = build_eval_spec(args)

    def __model_factory__(labels_list: list[str], embed_dim: int):
        return MoEJEPATextClassifier(
            embed_dim=embed_dim,
            predictor_hidden_dim=args.predictor_hidden,
            moe_num_experts=args.moe_num_experts,
            sigreg_weight=args.sigreg_weight,
            sigreg_num_slices=args.sigreg_num_slices,
            sigreg_lmbd=args.sigreg_lmbd,
        )

    extra = __extra_moe_eval_metrics__ if args.report_repr_metrics else None
    run_text_classification_eval(
        model_factory=__model_factory__,
        data=data,
        eval=eval_spec,
        checkpoint=args.checkpoint,
        strict=not getattr(args, "no_strict", False),
        extra_eval_metrics=extra,
    )


if __name__ == "__main__":
    main()
