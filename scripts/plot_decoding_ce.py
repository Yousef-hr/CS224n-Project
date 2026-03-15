#!/usr/bin/env python3
"""
Comparison plot: Decoding cross-entropy (eval_decoded_ce) — Energy-based vs JEPA.
"""
import json
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EB_STATE = PROJECT_ROOT / "eb_trainer_state.json"
JEPA_STATE = PROJECT_ROOT / "jepa_trainer_state.json"
OUTPUT = PROJECT_ROOT / "plot_decoding_ce.pdf"


def load_eval_series(path: Path, key: str):
    with open(path) as f:
        data = json.load(f)
    steps, values = [], []
    for entry in data.get("log_history", []):
        if key not in entry:
            continue
        step = entry.get("step")
        if step is None:
            continue
        steps.append(step)
        values.append(entry[key])
    return steps, values


def main():
    eb_steps, eb_ce = load_eval_series(EB_STATE, "eval_decoded_ce")
    jepa_steps, jepa_ce = load_eval_series(JEPA_STATE, "eval_decoded_ce")

    plt.figure(figsize=(6, 4))
    plt.plot(eb_steps, eb_ce, label="Energy-based", color="C0", linewidth=1.5)
    plt.plot(jepa_steps, jepa_ce, label="JEPA", color="C1", linewidth=1.5)
    plt.xlabel("Training step")
    plt.ylabel("Decoding cross-entropy")
    plt.title("Decoding cross-entropy: Energy-based vs JEPA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    plt.close()
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
