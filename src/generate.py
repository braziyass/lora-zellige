#!/usr/bin/env python
"""Génère les images d'évaluation : modèle de base + chaque checkpoint LoRA,
mêmes prompts et mêmes graines (comparaison équitable).

Sorties : results/generations/<config>/<prompt_id>_s<seed>.png + index.csv

Usage :
    python -m src.generate --config configs/base.yaml --include_base
    python -m src.generate --lora outputs/loras/r8/step_1000 --scale 0.8
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from src.utils import (fill_prompt, list_lora_checkpoints, load_config, parse_ckpt_name,
                       read_jsonl)

ADAPTER = "domain"


def load_pipeline(base: str):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def load_prompts(path: str, test_dir: str | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))
    # Légendes des images de test : mesure la généralisation à des descriptions jamais vues
    if test_dir and (Path(test_dir) / "metadata.jsonl").exists():
        for i, item in enumerate(read_jsonl(Path(test_dir) / "metadata.jsonl")):
            prompts.append({"id": f"t{i:02d}", "category": "held_out",
                            "prompt": item["text"], "ref_image": item["file_name"]})
    return prompts


def run_config(pipe, name, prompts, seeds, trigger, gcfg, out_dir, meta, writer, scale):
    cdir = out_dir / name
    cdir.mkdir(parents=True, exist_ok=True)
    for p in prompts:
        # Le prompt test contient déjà le trigger ; pour la base on le retire.
        tmpl = p["prompt"].replace(trigger, "{trigger}") if trigger else p["prompt"]
        text = fill_prompt(tmpl, None if meta["experiment"] == "base" else trigger)
        for s in seeds:
            fpath = cdir / f"{p['id']}_s{s}.png"
            if not fpath.exists():
                g = torch.Generator(device=pipe.device).manual_seed(int(s))
                t0 = time.time()
                img = pipe(text, num_inference_steps=gcfg["steps"], guidance_scale=gcfg["guidance"],
                           generator=g).images[0]
                img.save(fpath)
                dt = time.time() - t0
            else:
                dt = ""
            writer.writerow({"file": str(fpath), "config": name, "experiment": meta["experiment"],
                             "rank": meta["rank"] if meta["rank"] is not None else "",
                             "step": meta["step"], "scale": scale, "prompt_id": p["id"],
                             "category": p["category"], "prompt": text, "seed": s,
                             "gen_time_s": round(dt, 2) if dt != "" else ""})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--prompts", default="prompts/eval_prompts.csv")
    ap.add_argument("--lora", default=None, help="un seul dossier step_<n> ; sinon tous les checkpoints")
    ap.add_argument("--steps_filter", type=int, nargs="*", default=None, help="ex. 500 1500")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--include_base", action="store_true")
    ap.add_argument("--base_only", action="store_true", help="baseline : modèle de base seul")
    ap.add_argument("--out", default="results/generations")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    gcfg = cfg["generation"]
    scale = args.scale if args.scale is not None else gcfg["lora_scale"]
    trigger = cfg["data"]["trigger"]
    prompts = load_prompts(args.prompts, cfg["data"]["test_dir"])
    seeds = gcfg["seeds"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = [Path(args.lora)] if args.lora else list_lora_checkpoints(cfg["output_dir"])
    if args.base_only:
        ckpts, args.include_base = [], True
    if args.steps_filter:
        ckpts = [c for c in ckpts if parse_ckpt_name(c)["step"] in args.steps_filter]
    print(f"{len(prompts)} prompts × {len(seeds)} graines × {len(ckpts) + int(args.include_base)} configurations")

    pipe = load_pipeline(cfg["model"]["pretrained"])
    index_path = out_dir / "index.csv"
    new_file = not index_path.exists()
    fields = ["file", "config", "experiment", "rank", "step", "scale", "prompt_id",
              "category", "prompt", "seed", "gen_time_s"]
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        if args.include_base:
            print("→ modèle de base")
            run_config(pipe, "base", prompts, seeds, trigger, gcfg, out_dir,
                       {"experiment": "base", "rank": 0, "step": 0}, w, 0.0)
        for c in ckpts:
            meta = parse_ckpt_name(c)
            name = f"{meta['experiment']}_step{meta['step']}_x{scale}"
            print(f"→ {name}")
            pipe.load_lora_weights(str(c), adapter_name=ADAPTER)
            pipe.set_adapters([ADAPTER], adapter_weights=[scale])
            run_config(pipe, name, prompts, seeds, trigger, gcfg, out_dir, meta, w, scale)
            pipe.unload_lora_weights()
    print(f"Index : {index_path}  (si vous relancez, dédupliquez avec drop_duplicates('file'))")


if __name__ == "__main__":
    main()
