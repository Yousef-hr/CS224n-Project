from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from utils.encoders import OpenCLIPTextEncoder
from utils.metrics import covariance_spectrum, effective_rank, variance_ratio
from utils.train import load_checkpoint, setup_device

from .data import load_text_classification
from .base import DataSpec, EvalSpec, RunContext, TextClassificationModel
from .train import __collate_text__, __build_label_embeddings__, __encode_batch__

ExtraEvalFn = Callable[[TextClassificationModel, DataLoader, RunContext], dict[str, float]]
ModelFactory = Callable[..., TextClassificationModel]  # (labels_list, embed_dim, **kwargs) or (labels_list)


@torch.no_grad()
def run_text_classification_eval(
    *,
    model_factory: ModelFactory,
    data: DataSpec,
    eval: EvalSpec,
    extra_eval_metrics: ExtraEvalFn | None = None,
    checkpoint: str | Path | None = None,
    strict: bool = True,
) -> dict[str, float]:
    device = setup_device(eval.device)

    train_ds, test_ds, get_labels = load_text_classification(
        name=data.dataset,
        subset=data.subset,
        cache_dir=data.cache_dir,
    )
    labels_list = get_labels(train_ds.split)

    encoder = OpenCLIPTextEncoder(name=data.clip_model, pretrained=data.clip_pretrained)
    encoder.to(device)
    encoder.eval()
    embed_dim = encoder.embed_dim

    model = model_factory(labels_list, embed_dim)

    if isinstance(model, torch.nn.Module):
        model.to(device)

    if checkpoint is not None:
        load_checkpoint(Path(checkpoint), model, device=device, strict=strict)

    test_loader = DataLoader(
        test_ds,
        batch_size=data.batch_size,
        shuffle=False,
        collate_fn=__collate_text__,
        num_workers=data.num_workers,
    )

    use_amp = device.type == "cuda"
    label_embeddings = __build_label_embeddings__(encoder, labels_list, device, use_amp)
    ctx = RunContext(
        device=device,
        use_amp=use_amp,
        label_embeddings=label_embeddings,
        labels_list=labels_list,
        extra={"encoder": encoder},
    )

    model.eval_mode()

    all_preds: list[int] = []
    all_labels: list[int] = []
    correct = 0
    total = 0

    # Optional representation diagnostics (embedding-native outputs)
    pred_emb_all: list[torch.Tensor] = []
    target_emb_all: list[torch.Tensor] = []

    for inputs, labels in test_loader:
        labels = labels.to(device)
        input_emb = __encode_batch__(encoder, inputs, device, use_amp)
        outputs = model.forward(input_emb, ctx)
        scores = model.scores(outputs, ctx)
        pred = scores.argmax(dim=1)

        all_preds.extend(pred.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))

        if eval.report_repr_metrics and ctx.label_embeddings is not None:
            # Heuristic: if outputs are [B, D] matching label embedding dim, treat as pred embeddings.
            if outputs.ndim == 2 and outputs.size(1) == ctx.label_embeddings.size(1):
                pred_emb = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
                pred_emb_all.append(pred_emb.detach().cpu())
                target_emb_all.append(ctx.label_embeddings[labels].detach().cpu())

    acc = correct / max(total, 1)
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")

    idx_to_label = {i: lab for i, lab in enumerate(labels_list)}
    n_classes = len(labels_list)
    class_correct = [0] * n_classes
    class_total = [0] * n_classes
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    print("\nPer-class accuracy (top 15 by support):")
    sorted_by_support = sorted(range(n_classes), key=lambda i: class_total[i], reverse=True)
    for i in sorted_by_support[:15]:
        c_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"  {idx_to_label[i]:40s}: {c_acc:.4f} ({class_correct[i]}/{class_total[i]})")

    out: dict[str, float] = {"eval_acc": float(acc)}

    if eval.report_repr_metrics and pred_emb_all and target_emb_all:
        pred = torch.cat(pred_emb_all, dim=0)
        tgt = torch.cat(target_emb_all, dim=0)
        pred_spectrum = covariance_spectrum(pred)
        tgt_spectrum = covariance_spectrum(tgt)
        topk = min(eval.repr_topk_eigs, int(pred_spectrum.numel()))

        out.update(
            {
                "pred_effective_rank": effective_rank(pred),
                "target_effective_rank": effective_rank(tgt),
                "variance_ratio": variance_ratio(pred, tgt),
                "pred_cov_topk_eigs": float(pred_spectrum[:topk].mean().item()),
                "target_cov_topk_eigs": float(tgt_spectrum[:topk].mean().item()),
            }
        )

        print("\nRepresentation metrics:")
        print(f"  predictor effective rank: {out['pred_effective_rank']:.4f}")
        print(f"  target effective rank:    {out['target_effective_rank']:.4f}")
        print(f"  variance ratio (pred/target): {out['variance_ratio']:.4f}")
        print(f"  predictor covariance top-{topk} eigvals: {[round(v, 6) for v in pred_spectrum[:topk].tolist()]}")
        print(f"  target covariance top-{topk} eigvals:    {[round(v, 6) for v in tgt_spectrum[:topk].tolist()]}")

    if extra_eval_metrics is not None:
        out.update(extra_eval_metrics(model, test_loader, ctx))

    return out

