"""Jeu de données image + légende au format metadata.jsonl
({"file_name": "...", "text": "..."}), compatible HF imagefolder."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import read_jsonl


class CaptionedImageDataset(Dataset):
    def __init__(self, root, tokenizer, resolution: int = 512, random_flip: bool = True):
        self.root = Path(root)
        meta = self.root / "metadata.jsonl"
        if not meta.exists():
            raise FileNotFoundError(f"{meta} introuvable — lancez d'abord src.prepare_data")
        self.items = read_jsonl(meta)
        self.tokenizer = tokenizer
        tf = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
        ]
        if random_flip:
            tf.append(transforms.RandomHorizontalFlip())
        tf += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.tf = transforms.Compose(tf)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        img = Image.open(self.root / item["file_name"]).convert("RGB")
        ids = self.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.tf(img), "input_ids": ids}
