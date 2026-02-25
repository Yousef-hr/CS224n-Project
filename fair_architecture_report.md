# Fair Architecture Report (Yahoo Answers)

This report documents the default architecture alignment applied for fairer comparison across all model variants.

## Shared defaults

- Dataset: `yahoo_answers`
- Encoder: `ViT-B-32` + `laion2b_s34b_b79k` (frozen OpenCLIP text encoder)
- Batch size: `128`
- Epochs: `3`
- Learning rate: `3e-4`

## Head architecture alignment

- JEPA Baseline: `embed_dim -> 1024 -> embed_dim`
- JEPA + SIGReg: `embed_dim -> 1024 -> embed_dim`
- JEPA MoE: each expert uses hidden size `1024`, `4` experts
- JEPA MoE + SIGReg: each expert uses hidden size `1024`, `4` experts
- SOTA Baseline: classifier head `embed_dim -> 1024 -> num_classes`

## Why this is fairer

- All non-MoE variants now use one hidden layer of width `1024`.
- MoE variants keep the MoE mechanism but match per-expert hidden width (`1024`) and keep a fixed expert count (`4`).
- Optimization defaults are aligned across all tracks so performance differences are more attributable to method design (JEPA/SIGReg/MoE) instead of mismatched default capacity or learning rate.

## Notes

- Method-specific components remain intentionally different:
  - SIGReg variants include regularization terms.
  - MoE variants include routing and expert aggregation.
- Training CSV logging remains enabled in all train scripts via `--metrics_csv` (or their default checkpoint CSV paths).
