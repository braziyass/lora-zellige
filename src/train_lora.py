#!/usr/bin/env python
"""Entraînement d'une LoRA sur le UNet de Stable Diffusion 1.5.

Budget mémoire ≤ 12 Go : poids gelés en fp16, paramètres LoRA en fp32,
gradient checkpointing, batch 1 → ~6-7 Go de VRAM sur T4.

Sorties dans outputs/loras/<experiment_name>/ :
    step_<n>/pytorch_lora_weights.safetensors   (un point par save_steps)
    train_log.csv                               (perte, lr, temps)
    run_info.json                               (coût : temps, VRAM, nb paramètres, taille)

Usage :
    python -m src.train_lora --config configs/base.yaml
    python -m src.train_lora --config configs/base.yaml --set lora.rank=16 lora.alpha=16 experiment_name=r16
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import CaptionedImageDataset
from src.utils import disable_torchao, load_config, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur")
    return p.parse_args()


def save_lora(unet, path: Path) -> float:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    path.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(path), unet_lora_layers=state, safe_serialization=True
    )
    return (path / "pytorch_lora_weights.safetensors").stat().st_size / 1e6


def train(cfg: dict) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    disable_torchao()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg, lcfg, dcfg = cfg["train"], cfg["lora"], cfg["data"]
    use_fp16 = tcfg["mixed_precision"] == "fp16" and device == "cuda"
    wdtype = torch.float16 if use_fp16 else torch.float32

    base = cfg["model"]["pretrained"]
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    # --- Ajout de la LoRA : seules ces petites matrices A·B sont entraînées ---
    unet.add_adapter(LoraConfig(
        r=lcfg["rank"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        init_lora_weights="gaussian", target_modules=lcfg["target_modules"],
    ))
    for m in (text_encoder, vae, unet):
        m.to(device, dtype=wdtype)
    lora_params = []
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # LoRA en fp32 pour la stabilité
            lora_params.append(p)
    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"Paramètres entraînables : {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.3f} % du UNet)")

    if tcfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    unet.train()

    ds = CaptionedImageDataset(dcfg["train_dir"], tokenizer, dcfg["resolution"], dcfg["random_flip"])
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=tcfg["num_workers"], drop_last=True)
    print(f"{len(ds)} images d'entraînement")

    optimizer = torch.optim.AdamW(lora_params, lr=tcfg["lr"], weight_decay=1e-2)
    max_steps = tcfg["max_steps"]
    lr_sched = get_scheduler(tcfg["lr_scheduler"], optimizer,
                             num_warmup_steps=tcfg["warmup_steps"], num_training_steps=max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    accum = tcfg["grad_accum"]

    out_root = Path(cfg["output_dir"]) / cfg["experiment_name"]
    out_root.mkdir(parents=True, exist_ok=True)
    save_steps = set(tcfg["save_steps"]) | {max_steps}
    log_rows, ckpt_sizes, ckpt_times = [], {}, {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step, micro, t0 = 0, 0, time.time()
    running = 0.0
    while step < max_steps:
        for batch in loader:
            pixels = batch["pixel_values"].to(device, dtype=wdtype)
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                cond = text_encoder(ids)[0]
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, t)
            else:
                target = noise

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                pred = unet(noisy, t, cond).sample
            loss = F.mse_loss(pred.float(), target.float()) / accum
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1
            if micro % accum:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % tcfg["log_every"] == 0 or step == 1:
                elapsed = time.time() - t0
                log_rows.append({"step": step, "loss": round(running, 5),
                                 "lr": lr_sched.get_last_lr()[0], "time_s": round(elapsed, 1)})
                print(f"step {step:5d}/{max_steps}  loss {running:.4f}  {elapsed / 60:.1f} min")
            running = 0.0

            if step in save_steps:
                ckpt_sizes[step] = save_lora(unet, out_root / f"step_{step}")
                ckpt_times[step] = round(time.time() - t0, 1)
                print(f"  ↳ sauvegarde step_{step} ({ckpt_sizes[step]:.1f} Mo)")
            if step >= max_steps:
                break

    total = time.time() - t0
    with open(out_root / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "time_s"])
        w.writeheader()
        w.writerows(log_rows)

    info = {
        "experiment": cfg["experiment_name"],
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "n_train_images": len(ds),
        "train_time_min": round(total / 60, 2),
        "time_to_step_s": ckpt_times,
        "steps_per_s": round(max_steps / total, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
        "trainable_params": n_trainable,
        "trainable_pct_unet": round(100 * n_trainable / n_total, 4),
        "lora_size_mb": ckpt_sizes,
        "config": cfg,
    }
    save_json(info, out_root / "run_info.json")
    print(f"Terminé en {info['train_time_min']} min — pic VRAM {info['peak_vram_gb']} Go")
    return info


def main():
    args = parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
