#!/usr/bin/env python
"""Balayage des rangs LoRA (les étapes sont comparées via les checkpoints).

Usage :
    python -m src.sweep --sweep configs/sweep.yaml
    python -m src.sweep --sweep configs/sweep.yaml --ranks 4 16   # sous-ensemble
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import yaml

from src.train_lora import train
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--ranks", type=int, nargs="*", default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    sw = yaml.safe_load(open(args.sweep, encoding="utf-8"))
    ranks = args.ranks or sw["ranks"]
    for r in ranks:
        alpha = r if sw.get("alpha_equals_rank", True) else load_config(sw["base_config"])["lora"]["alpha"]
        overrides = [
            f"experiment_name=r{r}", f"lora.rank={r}", f"lora.alpha={alpha}",
            f"train.max_steps={sw['max_steps']}", f"train.save_steps={sw['save_steps']}",
        ]
        cfg = load_config(sw["base_config"], overrides)
        done = Path(cfg["output_dir"]) / cfg["experiment_name"] / "run_info.json"
        if args.skip_existing and done.exists():
            print(f"[rang {r}] déjà entraîné → ignoré")
            continue
        print(f"\n===== Rang {r} (alpha={alpha}) =====")
        train(cfg)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
