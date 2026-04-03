"""Generic training runner for vision QA (shared across model variants).

Provides two paths:
  1. ``run_vision_qa_train``          – encode on-the-fly with the model's CLIP
  2. ``run_vision_qa_train_cached``   – use precomputed CLIP embeddings
  3. ``prepare_and_run_cached``       – one-call helper: precompute ➜ release
     encoder ➜ train with cache
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train import setup_device, setup_seed, save_checkpoint, write_metrics_csv

from .base import DataSpec, RunContext, TrainSpec, VisionQABatch, VisionQAModel
from .data import load_vision_qa, make_collate_vision_qa


def _mean_reduce(metric_sums: dict[str, float], n: int) -> dict[str, float]:
    return {k: (v / max(n, 1)) for k, v in metric_sums.items()}


def _accumulate_metrics(
    sums: dict[str, float], metrics: dict[str, torch.Tensor], weight: int
) -> None:
    for k, v in metrics.items():
        if not torch.is_tensor(v):
            continue
        sums[k] = sums.get(k, 0.0) + float(v.detach().item()) * weight


def train_epoch(
    model: VisionQAModel,
    loader: Iterable[VisionQABatch],
    ctx: RunContext,
) -> dict[str, float]:
    model.train_mode()
    sums: dict[str, float] = {}
    n = 0

    for batch in tqdm(loader, desc="Train"):
        batch.answer_indices = batch.answer_indices.to(ctx.device)
        batch.images = batch.images.to(ctx.device)
        with torch.amp.autocast(
            device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
        ):
            embeddings = model.encode_inputs(batch, ctx)
            outputs = model.forward(embeddings, ctx)
            loss_dict = model.loss(outputs, batch.answer_indices, ctx, batch=batch)
            loss = loss_dict["total_loss"]

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        bs = int(batch.answer_indices.size(0))
        n += bs
        _accumulate_metrics(sums, loss_dict, bs)

    return _mean_reduce(sums, n)


@torch.no_grad()
def eval_epoch(
    model: VisionQAModel,
    loader: Iterable[VisionQABatch],
    ctx: RunContext,
) -> dict[str, float]:
    model.eval_mode()
    sums: dict[str, float] = {}
    correct = 0
    n = 0

    for batch in tqdm(loader, desc="Eval"):
        batch.answer_indices = batch.answer_indices.to(ctx.device)
        batch.images = batch.images.to(ctx.device)
        embeddings = model.encode_inputs(batch, ctx)
        outputs = model.forward(embeddings, ctx)

        loss_dict = model.loss(outputs, batch.answer_indices, ctx, batch=batch)
        scores = model.scores(outputs, ctx)

        pred = scores.argmax(dim=1)
        correct += int((pred == batch.answer_indices).sum().item())
        bs = int(batch.answer_indices.size(0))
        n += bs
        _accumulate_metrics(sums, loss_dict, bs)

    out = _mean_reduce(sums, n)
    out["eval_acc"] = correct / max(n, 1)
    return out


ExtraEvalMetricsFn = Callable[
    [VisionQAModel, DataLoader[VisionQABatch], RunContext], dict[str, float]
]


def run_vision_qa_train(
    *,
    model: VisionQAModel,
    data: DataSpec,
    train: TrainSpec,
    image_transform: Callable[[Any], torch.Tensor] | None = None,
    extra_eval_metrics: ExtraEvalMetricsFn | None = None,
    save_name: str = "best.pt",
) -> dict[str, float]:
    """
    Generic training loop for vision QA.
    Uses load_vision_qa() for train/val/test; eval is run on validation split.
    """
    device = setup_device(train.device)
    setup_seed(train.seed)
    save_dir = Path(train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, get_labels = load_vision_qa(
        name=data.dataset,
        subset=data.subset,
        cache_dir=data.cache_dir,
        use_image=data.use_image,
    )
    get_labels(train_ds.split)  # for registry compatibility; unused

    collate_fn = make_collate_vision_qa(image_transform=image_transform)

    train_loader: DataLoader[VisionQABatch] = DataLoader(
        train_ds,
        batch_size=data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=data.num_workers,
    )
    val_loader: DataLoader[VisionQABatch] = DataLoader(
        val_ds,
        batch_size=data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data.num_workers,
    )

    if isinstance(model, torch.nn.Module):
        model.to(device)

    use_amp = device.type == "cuda"
    ctx = RunContext(device=device, use_amp=use_amp, extra=None)

    metrics_rows: list[dict[str, float]] = []
    metrics_csv_path = (
        Path(train.metrics_csv)
        if train.metrics_csv is not None
        else (save_dir / "training_metrics.csv")
    )

    best_acc = -1.0
    best_summary: dict[str, float] = {}
    for ep in range(train.epochs):
        train_metrics = train_epoch(model=model, loader=train_loader, ctx=ctx)
        eval_metrics = eval_epoch(model=model, loader=val_loader, ctx=ctx)

        extra: dict[str, float] = {}
        if extra_eval_metrics is not None:
            extra = extra_eval_metrics(model, val_loader, ctx)

        row = {
            "epoch": float(ep + 1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **eval_metrics,
            **extra,
        }
        metrics_rows.append(row)

        acc = float(eval_metrics["eval_acc"])
        print(
            f"Epoch {ep + 1}/{train.epochs} | "
            f"train_total_loss={train_metrics.get('total_loss', 0.0):.4f} | "
            f"eval_total_loss={eval_metrics.get('total_loss', 0.0):.4f} | "
            f"eval_acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            best_summary = {**row}
            save_checkpoint(
                save_dir / save_name,
                model=model,
                optimizer=model.optimizer,
                epoch=ep,
                eval_acc=acc,
            )
            print(f"  Saved best checkpoint: {save_dir / save_name} (acc={acc:.4f})")

    write_metrics_csv(metrics_csv_path, metrics_rows)
    print(f"Saved training metrics CSV: {metrics_csv_path}")

    return {
        "best_eval_acc": best_acc,
        **{
            k: float(v)
            for k, v in best_summary.items()
            if isinstance(v, (int, float))
        },
    }


# ======================================================================
# Cached-embedding variants (skip CLIP at train time)
# ======================================================================

def train_epoch_cached(
    model: VisionQAModel,
    loader: Iterable,
    ctx: RunContext,
) -> dict[str, float]:
    model.train_mode()
    sums: dict[str, float] = {}
    n = 0

    for fused_emb, choice_embs, answer_indices, num_choices in tqdm(loader, desc="Train"):
        fused_emb = fused_emb.to(ctx.device, dtype=torch.float32)
        choice_embs = choice_embs.to(ctx.device, dtype=torch.float32)
        answer_indices = answer_indices.to(ctx.device)
        num_choices = num_choices.to(ctx.device)

        model._choice_embs = choice_embs
        model._num_choices = num_choices

        with torch.amp.autocast(
            device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
        ):
            outputs = model.forward(fused_emb, ctx)
            loss_dict = model.loss(outputs, answer_indices, ctx)
            loss = loss_dict["total_loss"]

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        bs = int(answer_indices.size(0))
        n += bs
        _accumulate_metrics(sums, loss_dict, bs)

    return _mean_reduce(sums, n)


@torch.no_grad()
def eval_epoch_cached(
    model: VisionQAModel,
    loader: Iterable,
    ctx: RunContext,
) -> dict[str, float]:
    model.eval_mode()
    sums: dict[str, float] = {}
    correct = 0
    n = 0

    for fused_emb, choice_embs, answer_indices, num_choices in tqdm(loader, desc="Eval"):
        fused_emb = fused_emb.to(ctx.device, dtype=torch.float32)
        choice_embs = choice_embs.to(ctx.device, dtype=torch.float32)
        answer_indices = answer_indices.to(ctx.device)
        num_choices = num_choices.to(ctx.device)

        model._choice_embs = choice_embs
        model._num_choices = num_choices

        outputs = model.forward(fused_emb, ctx)
        loss_dict = model.loss(outputs, answer_indices, ctx)
        scores = model.scores(outputs, ctx)

        pred = scores.argmax(dim=1)
        correct += int((pred == answer_indices).sum().item())
        bs = int(answer_indices.size(0))
        n += bs
        _accumulate_metrics(sums, loss_dict, bs)

    out = _mean_reduce(sums, n)
    out["eval_acc"] = correct / max(n, 1)
    return out


CachedExtraEvalMetricsFn = Callable[
    [VisionQAModel, DataLoader, RunContext], dict[str, float]
]


def run_vision_qa_train_cached(
    *,
    model: VisionQAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_spec: TrainSpec,
    extra_eval_metrics: CachedExtraEvalMetricsFn | None = None,
    save_name: str = "best.pt",
) -> dict[str, float]:
    """Training loop using precomputed CLIP embeddings (no encoder at runtime)."""
    device = setup_device(train_spec.device)
    setup_seed(train_spec.seed)
    save_dir = Path(train_spec.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        model.to(device)

    use_amp = device.type == "cuda"
    ctx = RunContext(device=device, use_amp=use_amp, extra=None)

    metrics_rows: list[dict[str, float]] = []
    metrics_csv_path = (
        Path(train_spec.metrics_csv)
        if train_spec.metrics_csv is not None
        else (save_dir / "training_metrics.csv")
    )

    best_acc = -1.0
    best_summary: dict[str, float] = {}
    for ep in range(train_spec.epochs):
        train_metrics = train_epoch_cached(model=model, loader=train_loader, ctx=ctx)
        eval_metrics = eval_epoch_cached(model=model, loader=val_loader, ctx=ctx)

        extra: dict[str, float] = {}
        if extra_eval_metrics is not None:
            try:
                extra = extra_eval_metrics(model, val_loader, ctx)
            except Exception as e:
                print(f"  Warning: extra eval metrics failed: {e}")

        row = {
            "epoch": float(ep + 1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **eval_metrics,
            **extra,
        }
        metrics_rows.append(row)

        acc = float(eval_metrics["eval_acc"])
        print(
            f"Epoch {ep + 1}/{train_spec.epochs} | "
            f"train_total_loss={train_metrics.get('total_loss', 0.0):.4f} | "
            f"eval_total_loss={eval_metrics.get('total_loss', 0.0):.4f} | "
            f"eval_acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            best_summary = {**row}
            save_checkpoint(
                save_dir / save_name,
                model=model,
                optimizer=model.optimizer,
                epoch=ep,
                eval_acc=acc,
            )
            print(f"  Saved best checkpoint: {save_dir / save_name} (acc={acc:.4f})")

        write_metrics_csv(metrics_csv_path, metrics_rows)

    print(f"Saved training metrics CSV: {metrics_csv_path}")

    return {
        "best_eval_acc": best_acc,
        **{
            k: float(v)
            for k, v in best_summary.items()
            if isinstance(v, (int, float))
        },
    }


def prepare_and_run_cached(
    *,
    model,
    dataset_name: str,
    subset: str | None,
    hf_cache_dir: str | None,
    use_image: bool,
    embedding_cache_dir: str,
    clip_model_name: str,
    clip_pretrained: str,
    precompute_batch_size: int,
    batch_size: int,
    train_spec: TrainSpec,
    extra_eval_metrics: CachedExtraEvalMetricsFn | None = None,
    save_name: str = "best.pt",
) -> dict[str, float]:
    """One-call helper: precompute → release encoder → train with cache."""
    from utils.vqa_embedding_cache import (
        build_cached_vqa_loaders,
        build_vqa_embedding_cache,
        load_vqa_split_cache,
        vqa_cache_exists,
    )

    cache_dataset_id = f"{dataset_name}__{subset}" if subset else dataset_name

    if not vqa_cache_exists(embedding_cache_dir, cache_dataset_id, clip_model_name, clip_pretrained):
        print("Building VQA embedding cache …")
        train_ds, val_ds, test_ds, _ = load_vision_qa(
            name=dataset_name, subset=subset, cache_dir=hf_cache_dir, use_image=use_image,
        )
        device = setup_device(train_spec.device)
        model.to(device)
        build_vqa_embedding_cache(
            cache_dir=embedding_cache_dir,
            dataset_id=cache_dataset_id,
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            model=model,
            splits={"train": train_ds, "validation": val_ds, "test": test_ds},
            image_transform=model.get_image_transform(),
            device=device,
            batch_size=precompute_batch_size,
        )
        model.cpu()
        del train_ds, val_ds, test_ds

    model.release_encoder()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_cache = load_vqa_split_cache(embedding_cache_dir, cache_dataset_id, clip_model_name, clip_pretrained, "train")
    val_cache = load_vqa_split_cache(embedding_cache_dir, cache_dataset_id, clip_model_name, clip_pretrained, "validation")
    train_loader, val_loader = build_cached_vqa_loaders(train_cache, val_cache, batch_size)

    return run_vision_qa_train_cached(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_spec=train_spec,
        extra_eval_metrics=extra_eval_metrics,
        save_name=save_name,
    )
