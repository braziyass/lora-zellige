#!/usr/bin/env python
"""Évaluation chiffrée des images générées.

Mesures par image :
  clip_score     respect du texte : 100 × cos(CLIP image, CLIP texte), trigger retiré
  domain_sim     fidélité au domaine : cos moyen (CLIP image) avec les images de TEST réelles
  train_nn_sim   copie / surapprentissage : similarité DINOv2 max avec une image d'ENTRAÎNEMENT
  is_copy        train_nn_sim > seuil (quasi-copie)
Mesures par (configuration, prompt) :
  diversity      1 − cos moyen DINOv2 entre graines (faible = effondrement des sorties)
Coût (depuis run_info.json) : temps d'entraînement jusqu'au checkpoint, VRAM, taille, paramètres.

Référence de copie : on calcule la même similarité max pour les images de test réelles
→ « niveau naturel » d'une image du domaine non vue. Un ratio > 1 signale de la mémorisation.

Usage :
    python -m src.evaluate --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.utils import build_generation_index, load_config, read_jsonl, save_json, strip_trigger


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(x, attr, None) is not None:
            return getattr(x, attr)
    return x[0]


class Embedder:
    def __init__(self, clip_id="openai/clip-vit-large-patch14", dino_id="facebook/dinov2-small"):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPModel.from_pretrained(clip_id).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(clip_id)
        self.dino = AutoModel.from_pretrained(dino_id).to(self.device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_id)

    @torch.no_grad()
    def clip_images(self, paths, bs=16):
        out = []
        for i in range(0, len(paths), bs):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            inp = self.clip_proc(images=imgs, return_tensors="pt").to(self.device)
            out.append(_as_tensor(self.clip.get_image_features(**inp)).float().cpu())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)

    @torch.no_grad()
    def clip_texts(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            inp = self.clip_proc(text=texts[i:i + bs], return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            out.append(_as_tensor(self.clip.get_text_features(**inp)).float().cpu())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)

    @torch.no_grad()
    def dino_images(self, paths, bs=16):
        out = []
        for i in range(0, len(paths), bs):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            inp = self.dino_proc(images=imgs, return_tensors="pt").to(self.device)
            out.append(self.dino(**inp).pooler_output.float().cpu())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)


def load_costs(lora_root: Path) -> pd.DataFrame:
    rows = []
    for info_path in lora_root.glob("*/run_info.json"):
        info = json.loads(info_path.read_text())
        for step, t in info["time_to_step_s"].items():
            rows.append({
                "experiment": info["experiment"], "step": int(step),
                "train_time_min": round(t / 60, 2),
                "peak_vram_gb": info["peak_vram_gb"],
                "trainable_params": info["trainable_params"],
                "lora_size_mb": round(info["lora_size_mb"].get(str(step), np.nan), 2),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--gen_dir", default="results/generations")
    ap.add_argument("--out", default="results")
    ap.add_argument("--copy_threshold", type=float, default=0.90)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    trigger = cfg["data"]["trigger"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Index reconstruit depuis les fichiers (index.csv peut être vidé par Google Drive)
    idx = pd.DataFrame(build_generation_index(args.gen_dir, cfg))
    if idx.empty:
        raise SystemExit(f"Aucune image trouvée dans {args.gen_dir} — lancez d'abord src.generate")
    idx.to_csv(out / "generations_index.csv", index=False)
    print(idx.groupby("config").size().to_string())
    train_dir, test_dir = Path(cfg["data"]["train_dir"]), Path(cfg["data"]["test_dir"])
    train_paths = [str(train_dir / r["file_name"]) for r in read_jsonl(train_dir / "metadata.jsonl")]
    test_paths = [str(test_dir / r["file_name"]) for r in read_jsonl(test_dir / "metadata.jsonl")]
    print(f"{len(idx)} images générées, {len(train_paths)} train, {len(test_paths)} test")

    emb = Embedder()
    gen_paths = idx["file"].tolist()
    g_clip = emb.clip_images(gen_paths)
    g_dino = emb.dino_images(gen_paths)
    t_clip = emb.clip_images(test_paths)
    tr_dino = emb.dino_images(train_paths)
    te_dino = emb.dino_images(test_paths)
    txt = emb.clip_texts([strip_trigger(p, trigger) for p in idx["prompt"]])

    idx["clip_score"] = (100 * (g_clip * txt).sum(-1).clamp(min=0)).numpy()
    idx["domain_sim"] = (g_clip @ t_clip.T).mean(-1).numpy()
    idx["train_nn_sim"] = (g_dino @ tr_dino.T).max(-1).values.numpy()
    idx["is_copy"] = idx["train_nn_sim"] > args.copy_threshold

    # Niveau naturel : images de test réelles vs entraînement
    ref = float((te_dino @ tr_dino.T).max(-1).values.mean())
    idx["memorization_ratio"] = idx["train_nn_sim"] / ref

    # Diversité entre graines pour un même (config, prompt)
    div = {}
    for key, grp in idx.groupby(["config", "prompt_id"]):
        if len(grp) < 2:
            continue
        e = g_dino[grp.index.values]
        sim = e @ e.T
        n = len(grp)
        div[key] = 1 - float((sim.sum() - n) / (n * (n - 1)))
    idx["diversity"] = [div.get((c, p), np.nan) for c, p in zip(idx["config"], idx["prompt_id"])]

    idx.to_csv(out / "metrics_per_image.csv", index=False)

    agg = (idx.groupby(["config", "experiment", "rank", "step", "scale", "category"])
              .agg(clip_score=("clip_score", "mean"), domain_sim=("domain_sim", "mean"),
                   train_nn_sim=("train_nn_sim", "mean"), copy_rate=("is_copy", "mean"),
                   memorization_ratio=("memorization_ratio", "mean"),
                   diversity=("diversity", "mean"), n=("file", "count"))
              .reset_index())
    overall = (idx.groupby(["config", "experiment", "rank", "step", "scale"])
                  .agg(clip_score=("clip_score", "mean"), domain_sim=("domain_sim", "mean"),
                       train_nn_sim=("train_nn_sim", "mean"), copy_rate=("is_copy", "mean"),
                       memorization_ratio=("memorization_ratio", "mean"),
                       diversity=("diversity", "mean"), n=("file", "count"))
                  .reset_index().assign(category="all"))
    summary = pd.concat([agg, overall], ignore_index=True)
    costs = load_costs(Path(cfg["output_dir"]))
    if not costs.empty:
        summary = summary.merge(costs, on=["experiment", "step"], how="left")
    summary = summary.round(4).sort_values(["category", "rank", "step"])
    summary.to_csv(out / "metrics_summary.csv", index=False)
    save_json({"reference_test_vs_train_sim": round(ref, 4),
               "copy_threshold": args.copy_threshold}, out / "metrics_reference.json")

    print(summary[summary["category"] == "all"].to_string(index=False))
    print(f"\nRéférence (test réel ↔ train) : {ref:.3f}")

    from src.plots import make_all
    make_all(out / "metrics_summary.csv", out / "metrics_per_image.csv", out / "figures")


if __name__ == "__main__":
    main()
