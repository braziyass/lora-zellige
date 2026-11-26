#!/usr/bin/env python
"""Télécharge des images d'une catégorie Wikimedia Commons en ne gardant
QUE les licences compatibles (CC0, domaine public, CC BY, CC BY-SA).

Chaque image est tracée dans data/sources.csv (auteur, licence, URL) avec une
colonne `caption` VIDE : l'équipe écrit elle-même une légende courte et
descriptive pour chaque image retenue (étape obligatoire, pas d'automatisation
aveugle).

Usage (Colab / Kaggle, accès réseau requis) :
    python data/download_commons.py --category "Zellige" --limit 80
    python data/download_commons.py --category "Zellige in Fez" --limit 40
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia exige un User-Agent identifiable : mettez l'adresse de l'équipe.
HEADERS = {"User-Agent": "LoRA-student-project/1.0 (contact: equipe@example.com)"}
OK_PREFIXES = ("cc0", "public domain", "pd", "cc by ", "cc by-sa", "cc-by")
BAD_MARKERS = ("nc", "nd", "non-commercial", "noderivs", "fair use")
FIELDS = ["file", "source_url", "page_url", "author", "license", "license_url",
          "commons_description", "caption", "keep", "notes"]


def clean_html(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


def license_ok(short_name: str) -> bool:
    s = short_name.lower().strip()
    if any(re.search(rf"\b{m}\b", s) for m in BAD_MARKERS):
        return False
    return s.startswith(OK_PREFIXES)


def iter_category(category: str, width: int):
    params = {
        "action": "query", "format": "json",
        "generator": "categorymembers", "gcmtitle": f"Category:{category}",
        "gcmtype": "file", "gcmlimit": 50,
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": width,
    }
    while True:
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        for page in data.get("query", {}).get("pages", {}).values():
            if page.get("imageinfo"):
                yield page
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="ex. 'Zellige'")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--min_side", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024, help="largeur de la vignette téléchargée")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sources", default="data/sources.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src_path = Path(args.sources)
    rows: dict[str, dict] = {}
    if src_path.exists():
        with open(src_path, encoding="utf-8") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}

    n_new = n_skip_lic = n_skip_size = 0
    for page in iter_category(args.category, args.width):
        if n_new >= args.limit:
            break
        info = page["imageinfo"][0]
        if not info.get("mime", "").startswith("image/") or info.get("mime") == "image/svg+xml":
            continue
        meta = info.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if not license_ok(lic):
            n_skip_lic += 1
            continue
        if min(info.get("width", 0), info.get("height", 0)) < args.min_side:
            n_skip_size += 1
            continue
        url = info.get("thumburl") or info["url"]
        ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
        fname = f"commons_{page['pageid']}{ext}"
        if fname in rows:
            continue
        img = requests.get(url, headers=HEADERS, timeout=60)
        img.raise_for_status()
        (out / fname).write_bytes(img.content)
        rows[fname] = {
            "file": fname,
            "source_url": info["url"],
            "page_url": info.get("descriptionurl", ""),
            "author": clean_html(meta.get("Artist", {}).get("value")),
            "license": lic,
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "commons_description": clean_html(meta.get("ImageDescription", {}).get("value"))[:300],
            "caption": "",
            "keep": "",
            "notes": "",
        }
        n_new += 1
        time.sleep(0.3)

    with open(src_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(f"{n_new} nouvelles images → {out}")
    print(f"Ignorées : {n_skip_lic} (licence), {n_skip_size} (trop petites)")
    print(f"Étape suivante : ouvrir {src_path}, trier (keep=1/0) et écrire une légende courte par image.")


if __name__ == "__main__":
    main()
