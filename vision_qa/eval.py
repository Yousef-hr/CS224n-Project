"""Generic eval runner for vision QA."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from utils.train import load_checkpoint, setup_device

from .base import DataSpec, EvalSpec, RunContext, VisionQABatch, VisionQAModel
from .data import load_vision_qa, make_collate_vision_qa


@torch.no_grad()
def run_vision_qa_eval(
    *,
    model: VisionQAModel,
    data: DataSpec,
    eval_spec: EvalSpec,
    image_transform: Callable[[Any], torch.Tensor] | None = None,
    checkpoint: str | Path | None = None,
    strict: bool = True,
    split: str = "test",
) -> dict[str, float]:
    """
    Run evaluation on validation or test split.
    Returns dict with eval_acc and optionally accuracy by subject/grade when report_breakdown=True.
    """
    device = setup_device(eval_spec.device)

    train_ds, val_ds, test_ds, _ = load_vision_qa(
        name=data.dataset,
        subset=data.subset,
        cache_dir=data.cache_dir,
        use_image=data.use_image,
    )
    eval_ds = val_ds if split == "validation" else test_ds

    if isinstance(model, torch.nn.Module):
        model.to(device)

    if checkpoint is not None:
        load_checkpoint(Path(checkpoint), model, device=device, strict=strict)

    collate_fn = make_collate_vision_qa(image_transform=image_transform)
    loader: DataLoader[VisionQABatch] = DataLoader(
        eval_ds,
        batch_size=data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data.num_workers,
    )

    use_amp = device.type == "cuda"
    ctx = RunContext(device=device, use_amp=use_amp, extra=None)

    model.eval_mode()

    correct = 0
    total = 0
    subject_correct: dict[str, int] = defaultdict(int)
    subject_total: dict[str, int] = defaultdict(int)
    grade_correct: dict[str, int] = defaultdict(int)
    grade_total: dict[str, int] = defaultdict(int)

    for batch in loader:
        batch.answer_indices = batch.answer_indices.to(device)
        batch.images = batch.images.to(device)
        embeddings = model.encode_inputs(batch, ctx)
        outputs = model.forward(embeddings, ctx)
        scores = model.scores(outputs, ctx)
        pred = scores.argmax(dim=1)

        n = int(batch.answer_indices.size(0))
        correct += int((pred == batch.answer_indices).sum().item())
        total += n

        if eval_spec.report_breakdown and batch.subject_list is not None and batch.grade_list is not None:
            for i in range(n):
                s = batch.subject_list[i]
                g = batch.grade_list[i]
                subject_total[s] += 1
                grade_total[g] += 1
                if pred[i].item() == batch.answer_indices[i].item():
                    subject_correct[s] += 1
                    grade_correct[g] += 1

    acc = correct / max(total, 1)
    print(f"{split.capitalize()} accuracy: {acc:.4f} ({correct}/{total})")

    out: dict[str, float] = {"eval_acc": float(acc)}

    if eval_spec.report_breakdown and subject_total:
        print("\nAccuracy by subject:")
        for s in sorted(subject_total.keys()):
            c, t = subject_correct[s], subject_total[s]
            a = c / t if t else 0.0
            print(f"  {s}: {a:.4f} ({c}/{t})")
            out[f"acc_subject_{s}"] = a
        print("\nAccuracy by grade:")
        for g in sorted(grade_total.keys()):
            c, t = grade_correct[g], grade_total[g]
            a = c / t if t else 0.0
            print(f"  {g}: {a:.4f} ({c}/{t})")
            out[f"acc_grade_{g}"] = a

    return out
