"""Figures pour le rapport : courbes rang × étapes, compromis, grilles visuelles."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

METRICS = {
    "clip_score": "CLIP-score (respect du texte) ↑",
    "domain_sim": "Similarité au domaine (images test) ↑",
    "memorization_ratio": "Ratio de mémorisation (↓, 1 = naturel)",
    "diversity": "Diversité entre graines ↑",
}


def curves(summary: pd.DataFrame, category: str, out_dir: Path):
    df = summary[summary["category"] == category]
    lora = df[df["experiment"] != "base"]
    base = df[df["experiment"] == "base"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4))
    for ax, (m, label) in zip(axes, METRICS.items()):
        for rank, grp in lora.groupby("rank"):
            g = grp.sort_values("step")
            ax.plot(g["step"], g[m], marker="o", label=f"rang {int(rank)}")
        if not base.empty and m != "memorization_ratio":
            ax.axhline(base[m].iloc[0], color="grey", ls="--", lw=1, label="base")
        if m == "memorization_ratio":
            ax.axhline(1.0, color="grey", ls=":", lw=1)
        ax.set_xlabel("étapes d'entraînement")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Prompts : {category}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"curves_{category}.png", dpi=150)
    plt.close(fig)


def tradeoff(summary: pd.DataFrame, out_dir: Path):
    """Compromis : apprendre le domaine (x) sans perdre le contrôle par le texte (y)."""
    df = summary[summary["category"] == "composition"]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for rank, grp in df.groupby("rank"):
        lab = "base" if rank == 0 else f"rang {int(rank)}"
        ax.scatter(grp["domain_sim"], grp["clip_score"], s=20 + grp["step"] / 20, label=lab, alpha=0.8)
        for _, r in grp.iterrows():
            ax.annotate(str(int(r["step"])), (r["domain_sim"], r["clip_score"]), fontsize=6)
    ax.set_xlabel("Similarité au domaine ↑")
    ax.set_ylabel("CLIP-score sur prompts de composition ↑")
    ax.set_title("Compromis fidélité au domaine / respect du texte\n(taille = nb d'étapes)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff.png", dpi=150)
    plt.close(fig)


def visual_grid(per_image: pd.DataFrame, prompt_id: str, seed: int, out_dir: Path, thumb=192):
    """Lignes = rangs, colonnes = étapes ; première case = modèle de base."""
    df = per_image[(per_image["prompt_id"] == prompt_id) & (per_image["seed"] == seed)]
    if df.empty:
        return
    lora = df[df["experiment"] != "base"]
    ranks = sorted(lora["rank"].unique())
    steps = sorted(lora["step"].unique())
    if not ranks:
        return
    fig, axes = plt.subplots(len(ranks), len(steps) + 1,
                             figsize=(1.9 * (len(steps) + 1), 1.9 * len(ranks)), squeeze=False)
    base = df[df["experiment"] == "base"]
    for i, r in enumerate(ranks):
        ax = axes[i][0]
        if i == 0 and not base.empty:
            ax.imshow(Image.open(base["file"].iloc[0]).resize((thumb, thumb)))
            ax.set_title("base", fontsize=8)
        ax.axis("off")
        for j, s in enumerate(steps):
            ax = axes[i][j + 1]
            row = lora[(lora["rank"] == r) & (lora["step"] == s)]
            if not row.empty:
                ax.imshow(Image.open(row["file"].iloc[0]).resize((thumb, thumb)))
            if i == 0:
                ax.set_title(f"{s} étapes", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"rang {int(r)}", fontsize=8)
    fig.suptitle(df["prompt"].iloc[-1][:90], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"grid_{prompt_id}_s{seed}.png", dpi=150)
    plt.close(fig)


def make_all(summary_csv, per_image_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    per_image = pd.read_csv(per_image_csv)
    for cat in summary["category"].unique():
        curves(summary, cat, out_dir)
    tradeoff(summary, out_dir)
    for pid in per_image["prompt_id"].unique()[:6]:
        visual_grid(per_image, pid, int(per_image["seed"].min()), out_dir)
    print(f"Figures → {out_dir}")
