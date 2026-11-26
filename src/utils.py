"""Fonctions communes : configuration, graine, E/S."""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import yaml


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Charge un YAML puis applique des surcharges « a.b=valeur »."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Surcharge invalide : {item!r} (attendu clé=valeur)")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fill_prompt(template: str, trigger: str | None) -> str:
    """Remplit {trigger}. Sans LoRA (trigger=None), le jeton rare est retiré
    pour que le modèle de base reçoive un prompt descriptif équitable."""
    text = template.replace("{trigger}", trigger or "")
    return re.sub(r"\s+", " ", text).strip(" ,")


def strip_trigger(text: str, trigger: str) -> str:
    """Retire le jeton rare (inconnu de CLIP) avant de calculer le CLIP-score."""
    return re.sub(r"\s+", " ", re.sub(rf"\b{re.escape(trigger)}\b", "", text)).strip(" ,")


def list_lora_checkpoints(root: str | Path) -> list[Path]:
    """outputs/loras/<exp>/step_<n>/pytorch_lora_weights.safetensors"""
    root = Path(root)
    found = sorted(root.glob("*/step_*/pytorch_lora_weights.safetensors"))
    return [p.parent for p in found]


def parse_ckpt_name(ckpt_dir: Path) -> dict:
    """Retrouve rang et étape depuis le chemin et run_info.json."""
    info_path = ckpt_dir.parent / "run_info.json"
    rank = None
    if info_path.exists():
        rank = json.loads(info_path.read_text())["config"]["lora"]["rank"]
    step = int(ckpt_dir.name.split("_")[-1])
    return {"experiment": ckpt_dir.parent.name, "rank": rank, "step": step}


def disable_torchao() -> None:
    """Colab/Kaggle préinstallent une vieille version de torchao ; peft >= 0.17
    lève alors une ImportError en créant la LoRA, alors qu'on ne s'en sert pas.
    On désactive la détection de torchao dans peft."""
    try:
        import peft.import_utils as iu
        iu.is_torchao_available = lambda *a, **k: False
        import peft.tuners.lora.torchao as lt
        lt.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
