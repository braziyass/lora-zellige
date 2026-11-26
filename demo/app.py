#!/usr/bin/env python
"""Démo interactive : écrire un prompt, régler l'intensité de la LoRA,
comparer avant / après avec la même graine.

Colab / Kaggle :
    python demo/app.py --share
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from src.generate import ADAPTER, load_pipeline
from src.utils import fill_prompt, list_lora_checkpoints, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config)
TRIGGER = cfg["data"]["trigger"]
CKPTS = {f"{p.parent.name} / {p.name}": p for p in list_lora_checkpoints(ROOT / cfg["output_dir"])}
if not CKPTS:
    raise SystemExit("Aucune LoRA trouvée dans outputs/loras — entraînez-en une d'abord.")
DEFAULT = next((k for k in CKPTS if k.endswith("step_1000")), next(iter(CKPTS)))

pipe = load_pipeline(cfg["model"]["pretrained"])
state = {"loaded": None}


def ensure_lora(name: str):
    if state["loaded"] != name:
        if state["loaded"] is not None:
            pipe.unload_lora_weights()
        pipe.load_lora_weights(str(CKPTS[name]), adapter_name=ADAPTER)
        state["loaded"] = name


def generate(prompt, lora_name, scale, steps, guidance, seed, compare):
    ensure_lora(lora_name)
    with_trigger = fill_prompt(prompt, TRIGGER)
    without = fill_prompt(prompt, None)
    kw = dict(num_inference_steps=int(steps), guidance_scale=float(guidance))

    pipe.enable_lora()
    pipe.set_adapters([ADAPTER], adapter_weights=[float(scale)])
    after = pipe(with_trigger, generator=torch.Generator(pipe.device).manual_seed(int(seed)), **kw).images[0]

    before = None
    if compare:
        pipe.disable_lora()
        before = pipe(without, generator=torch.Generator(pipe.device).manual_seed(int(seed)), **kw).images[0]
        pipe.enable_lora()
    info = f"Prompt LoRA : « {with_trigger} »\nPrompt base : « {without} »"
    return before, after, info


EXAMPLES = [
    ["{trigger} zellige, eight-pointed star pattern in blue and white tiles"],
    ["a ceramic coffee mug decorated with {trigger} zellige pattern, product photo"],
    ["a modern minimalist bathroom with a wall of {trigger} zellige tiles"],
    ["{trigger} zellige pattern in purple and gold colors"],
]

with gr.Blocks(title="LoRA Zellige — avant / après") as demo:
    gr.Markdown(
        "## LoRA texte-image : zellige marocain\n"
        f"Écrivez `{{trigger}}` là où le concept doit apparaître (remplacé par **{TRIGGER}**). "
        "Le modèle de base reçoit le même prompt sans le jeton rare.\n\n"
        "*Images générées par IA — ne pas présenter comme des photographies réelles.*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0][0], lines=3)
            lora = gr.Dropdown(list(CKPTS), value=DEFAULT, label="LoRA (rang / étapes)")
            scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Intensité de la LoRA")
            with gr.Accordion("Réglages", open=False):
                steps = gr.Slider(10, 50, value=30, step=1, label="Étapes de débruitage")
                guidance = gr.Slider(1, 12, value=7.5, step=0.5, label="Guidance (CFG)")
                seed = gr.Number(value=0, precision=0, label="Graine")
            compare = gr.Checkbox(value=True, label="Comparer avec le modèle de base")
            btn = gr.Button("Générer", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                out_before = gr.Image(label="Avant (modèle de base)")
                out_after = gr.Image(label="Après (avec LoRA)")
            info = gr.Textbox(label="Prompts utilisés", lines=2)
    gr.Examples(EXAMPLES, inputs=[prompt])
    btn.click(generate, [prompt, lora, scale, steps, guidance, seed, compare],
              [out_before, out_after, info])

if __name__ == "__main__":
    demo.launch(share=args.share)
