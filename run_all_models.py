import argparse
import subprocess
import time
from pathlib import Path


MODEL_SCRIPTS = {
    "TC-JEPA-Baseline": "models/TC-JEPA-Baseline/train.py",
    "TC-JEPA-SIGReg": "models/TC-JEPA-SIGReg/train.py",
    "TC-JEPA-MoE": "models/TC-JEPA-MoE/train.py",
    "TC-JEPA-MoE-SIGReg": "models/TC-JEPA-MoE-SIGReg/train.py",
    "TC-SOTA-Baseline": "models/TC-SOTA-Baseline/train.py",
}


def build_command(
    *,
    python_bin: str,
    script_path: str,
    dataset: str,
    epochs: int,
    seed: int,
    save_dir: Path,
    cache_dir: str | None,
    embedding_cache_dir: str | None,
    device: str,
    encoder: str,
    clip_model: str,
    clip_pretrained: str,
    precompute_batch_size: int,
) -> list[str]:
    cmd = [
        python_bin,
        script_path,
        "--dataset",
        dataset,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--save_dir",
        str(save_dir),
        "--device",
        device,
        "--encoder",
        encoder,
        "--clip_model",
        clip_model,
        "--clip_pretrained",
        clip_pretrained,
        "--precompute_batch_size",
        str(precompute_batch_size),
    ]
    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])
    if embedding_cache_dir:
        cmd.extend(["--embedding_cache_dir", embedding_cache_dir])
    return cmd


def run_command(cmd: list[str], dry_run: bool) -> tuple[float, bool]:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0.0, True
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode}, continuing to next run...")
        return elapsed, False
    return elapsed, True


def main():
    parser = argparse.ArgumentParser(description="Stage all model runs with shared frozen-encoder cache.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--dataset", type=str, default="yahoo_answers")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache dir")
    parser.add_argument("--embedding_cache_dir", type=str, default="precomputed_embeddings")
    parser.add_argument("--save_root", type=str, default="runs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encoder", type=str, choices=["openclip", "minilm"], default="openclip")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--precompute_batch_size", type=int, default=512)
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--full_epochs", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        plan = [("TC-JEPA-Baseline", args.seeds[0], args.smoke_epochs)]
    else:
        plan = []
        for model_name in MODEL_SCRIPTS:
            for seed in args.seeds:
                plan.append((model_name, seed, args.full_epochs))

    print(f"Planned runs: {len(plan)}")
    timings: list[tuple[str, int, int, float, bool]] = []
    for model_name, seed, epochs in plan:
        script_path = MODEL_SCRIPTS[model_name]
        save_dir = save_root / model_name / f"seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(
            python_bin=args.python_bin,
            script_path=script_path,
            dataset=args.dataset,
            epochs=epochs,
            seed=seed,
            save_dir=save_dir,
            cache_dir=args.cache_dir,
            embedding_cache_dir=args.embedding_cache_dir,
            device=args.device,
            encoder=args.encoder,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            precompute_batch_size=args.precompute_batch_size,
        )
        elapsed, success = run_command(cmd, args.dry_run)
        timings.append((model_name, seed, epochs, elapsed, success))

    print("\nRun summary:")
    for model_name, seed, epochs, elapsed, success in timings:
        status = "OK" if success else "FAILED"
        if args.dry_run:
            print(f"- {model_name} seed={seed} epochs={epochs}")
        else:
            print(f"- [{status}] {model_name} seed={seed} epochs={epochs} elapsed={elapsed/60:.2f} min")
    failed = sum(1 for *_, s in timings if not s)
    if failed:
        print(f"\n{failed}/{len(timings)} runs failed.")


if __name__ == "__main__":
    main()
