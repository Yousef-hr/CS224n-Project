# Run Staging Guide (Single Command)

This setup stages all runs through one launcher command and reuses frozen-encoder embeddings.

## 1) Smoke test (first estimate)

Runs only `TC-JEPA-Baseline` for 1 epoch on one seed:

```bash
python run_all_models.py --mode smoke --dataset yahoo_answers --device cuda
```

What this does on first run:
- Builds precomputed frozen text embeddings into `precomputed_embeddings/`
- Trains baseline for 1 epoch
- Saves checkpoints/CSV under `runs/TC-JEPA-Baseline/seed_42/`

On later runs, it reuses the same embedding cache.

## 2) Full staged sweep (5 models x 3 seeds)

```bash
python run_all_models.py --mode full --dataset yahoo_answers --seeds 42 43 44 --device cuda
```

This launches all 15 runs sequentially with shared embedding cache.

## 3) Dry-run check (no training)

```bash
python run_all_models.py --mode full --dry_run
```

Use this to verify generated commands before launching.
