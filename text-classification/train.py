"""Generic training runner for text classification (shared across model variants)."""

from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.encoders import OpenCLIPTextEncoder
from utils.embedding_cache import build_cached_loaders, get_or_build_text_embedding_cache
from utils.train import setup_device, setup_seed, save_checkpoint, write_metrics_csv

from .base import DataSpec, RunContext, TextClassificationModel, TrainSpec
from .data import load_text_classification


def __collate_text__(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def __dataset_cache_id__(dataset: str, subset: str | None) -> str:
    return dataset if subset is None else f"{dataset}_{subset}"


def __encode_batch__(encoder, inputs, device: torch.device, use_amp: bool) -> torch.Tensor:
    """Encode text batch or pass through tensor. Runner-owned encoding."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device=device, dtype=torch.float32)
    encoder.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            emb = encoder(inputs)
    return emb.to(device=device, dtype=torch.float32)


@torch.no_grad()
def __build_label_embeddings__(
    encoder: OpenCLIPTextEncoder,
    labels_list: list[str],
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    encoder.eval()
    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        label_emb = encoder(labels_list).to(device=device, dtype=torch.float32)
    label_emb = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-12)
    return label_emb


def __mean_reduce__(metric_sums: dict[str, float], n: int) -> dict[str, float]:
    return {k: (v / max(n, 1)) for k, v in metric_sums.items()}


def __accumulate_metrics__(
    sums: dict[str, float], metrics: dict[str, torch.Tensor], weight: int
) -> None:
    for k, v in metrics.items():
        if not torch.is_tensor(v):
            continue
        sums[k] = sums.get(k, 0.0) + float(v.detach().item()) * weight


def train_epoch(
    model: TextClassificationModel,
    loader: Iterable,
    ctx: RunContext,
    encoder: OpenCLIPTextEncoder,
) -> dict[str, float]:
    model.train_mode()
    sums: dict[str, float] = {}
    n = 0

    for inputs, labels in tqdm(loader, desc="Train"):
        labels = labels.to(ctx.device)
        input_emb = __encode_batch__(encoder, inputs, ctx.device, ctx.use_amp)
        with torch.amp.autocast(
            device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
        ):
            outputs = model.forward(input_emb, ctx)
            loss_dict = model.loss(outputs, labels, ctx)
            loss = loss_dict["total_loss"]

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        bs = int(labels.size(0))
        n += bs
        __accumulate_metrics__(sums, loss_dict, bs)

    return __mean_reduce__(sums, n)


@torch.no_grad()
def eval_epoch(
    model: TextClassificationModel,
    loader: Iterable,
    ctx: RunContext,
    encoder: OpenCLIPTextEncoder,
) -> dict[str, float]:
    model.eval_mode()
    sums: dict[str, float] = {}
    correct = 0
    n = 0

    for inputs, labels in tqdm(loader, desc="Eval"):
        labels = labels.to(ctx.device)
        input_emb = __encode_batch__(encoder, inputs, ctx.device, ctx.use_amp)
        outputs = model.forward(input_emb, ctx)

        loss_dict = model.loss(outputs, labels, ctx)
        scores = model.scores(outputs, ctx)

        pred = scores.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        bs = int(labels.size(0))
        n += bs
        __accumulate_metrics__(sums, loss_dict, bs)

    out = __mean_reduce__(sums, n)
    out["eval_acc"] = correct / max(n, 1)
    return out


ExtraMetricsFn = Callable[[TextClassificationModel, DataLoader, RunContext], dict[str, float]]
ModelFactory = Callable[..., TextClassificationModel]  # (labels_list, embed_dim, **kwargs) or (labels_list)

def run_text_classification_train(
    *,
    model_factory: ModelFactory,
    data: DataSpec,
    train: TrainSpec,
    extra_eval_metrics: ExtraMetricsFn | None = None,
    save_name: str = "best.pt",
) -> dict[str, float]:
    """
    Generic training loop for text classification.
    Uses load_text_classification(name, subset=..., cache_dir=...) -> (train_ds, test_ds, get_labels).
    """
    device = setup_device(train.device)
    setup_seed(train.seed)
    save_dir = Path(train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    trainable = [p for p in model.parameters() if p.requires_grad]
    model.optimizer = torch.optim.AdamW(trainable, lr=train.lr, weight_decay=train.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=data.batch_size,
        shuffle=True,
        collate_fn=__collate_text__,
        num_workers=data.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data.batch_size,
        shuffle=False,
        collate_fn=__collate_text__,
        num_workers=data.num_workers,
    )

    use_amp = device.type == "cuda"
    ctx = RunContext(device=device, use_amp=use_amp, label_embeddings=None, labels_list=labels_list, extra={"encoder": encoder})

    if data.embedding_cache_dir:
        dataset_id = __dataset_cache_id__(data.dataset, data.subset)
        cache_payload = get_or_build_text_embedding_cache(
            cache_dir=data.embedding_cache_dir,
            dataset_id=dataset_id,
            clip_model=data.clip_model,
            clip_pretrained=data.clip_pretrained,
            train_ds=train_ds,
            test_ds=test_ds,
            labels_list=labels_list,
            encoder=encoder,
            device=device,
            precompute_batch_size=data.precompute_batch_size,
        )

        train_loader, test_loader = build_cached_loaders(cache_payload, data.batch_size)
        label_embeddings = cache_payload["label_embeddings"].to(device=device, dtype=torch.float32)
        label_embeddings = label_embeddings / (label_embeddings.norm(dim=-1, keepdim=True) + 1e-12)

        ctx = RunContext(
            device=device,
            use_amp=use_amp,
            label_embeddings=label_embeddings,
            labels_list=labels_list,
            extra={"encoder": encoder},
        )
        print("Using precomputed frozen encoder embeddings for train/test splits.")
    else:
        label_embeddings = __build_label_embeddings__(encoder, labels_list, device, use_amp)
        ctx = RunContext(
            device=device,
            use_amp=use_amp,
            label_embeddings=label_embeddings,
            labels_list=labels_list,
            extra={"encoder": encoder},
        )

    metrics_rows: list[dict[str, float]] = []
    metrics_csv_path = (
        Path(train.metrics_csv)
        if train.metrics_csv is not None
        else (save_dir / "training_metrics.csv")
    )

    best_acc = -1.0
    best_summary: dict[str, float] = {}
    for ep in range(train.epochs):
        train_metrics = train_epoch(model=model, loader=train_loader, ctx=ctx, encoder=encoder)
        eval_metrics = eval_epoch(model=model, loader=test_loader, ctx=ctx, encoder=encoder)

        extra: dict[str, float] = {}
        if extra_eval_metrics is not None:
            extra = extra_eval_metrics(model, test_loader, ctx)

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
