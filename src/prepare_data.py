#!/usr/bin/env python
"""Prépare le mini-jeu de données :
  1. lit data/sources.csv (garde keep=1, licence autorisée, légende non vide) ;
  2. recadre au centre et redimensionne en 512×512 ;
  3. supprime les doublons / quasi-doublons (hachage perceptuel dHash) ;
  4. réserve `test_size` images pour tester la généralisation ;
  5. écrit data/processed/{train,test}/metadata.jsonl + un rapport.

Usage :
    python -m src.prepare_data --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageOps

from src.utils import load_config, save_json, write_jsonl


def dhash(img: Image.Image, size: int = 8) -> int:
    """Hachage par différence : deux images proches ont peu de bits différents."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.tobytes())  # mode L : 1 octet par pixel
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def square_resize(img: Image.Image, res: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(img, (res, res), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def license_ok(lic: str, allowed: list[str]) -> bool:
    s = lic.lower().strip()
    if any(x in s for x in ("-nc", " nc", "-nd", " nd")):
        return False
    return any(s.startswith(a) for a in allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dup_threshold", type=int, default=6, help="distance de Hamming max pour un doublon")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]

    raw = Path(d["raw_dir"])
    out = Path(d["processed_dir"])
    res = d["resolution"]
    rejected = {"keep=0": [], "licence": [], "legende_vide": [], "introuvable": [],
                "trop_petite": [], "doublon": []}

    with open(d["sources_csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept, hashes = [], []
    for r in rows:
        name = r["file"]
        if r.get("keep", "1").strip() in ("0", "non", "no"):
            rejected["keep=0"].append(name); continue
        if not license_ok(r.get("license", ""), d["allowed_licenses"]):
            rejected["licence"].append(name); continue
        if not r.get("caption", "").strip():
            rejected["legende_vide"].append(name); continue
        path = raw / name
        if not path.exists():
            rejected["introuvable"].append(name); continue
        img = Image.open(path)
        if min(img.size) < res * 0.75:
            rejected["trop_petite"].append(name); continue
        img = square_resize(img, res)
        h = dhash(img)
        if any(hamming(h, h2) <= args.dup_threshold for h2 in hashes):
            rejected["doublon"].append(name); continue
        hashes.append(h)
        kept.append((name, img, r))

    rng = random.Random(cfg["seed"])
    rng.shuffle(kept)
    n_test = min(d["test_size"], max(0, len(kept) - 5))
    splits = {"test": kept[:n_test], "train": kept[n_test:]}

    for split, items in splits.items():
        meta = []
        sdir = out / split
        sdir.mkdir(parents=True, exist_ok=True)
        for name, img, r in items:
            fname = Path(name).stem + ".png"
            img.save(sdir / fname)
            text = d["caption_template"].format(trigger=d["trigger"], caption=r["caption"].strip())
            meta.append({"file_name": fname, "text": text, "source": name,
                         "license": r.get("license", ""), "author": r.get("author", "")})
        write_jsonl(meta, sdir / "metadata.jsonl")

    report = {
        "n_sources": len(rows),
        "n_train": len(splits["train"]),
        "n_test": len(splits["test"]),
        "rejets": {k: len(v) for k, v in rejected.items()},
        "detail_rejets": rejected,
    }
    save_json(report, out / "report.json")
    print(f"train={report['n_train']}  test={report['n_test']}  rejets={report['rejets']}")
    if not 20 <= report["n_train"] + report["n_test"] <= 50:
        print("⚠️  Le sujet demande 20 à 50 images au total : ajustez le corpus.")


if __name__ == "__main__":
    main()
