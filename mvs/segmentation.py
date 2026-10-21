"""Semantic segmentation (SegFormer-B5, Cityscapes, 19 classes).

Pure inference: produces one class-id map per frame, no masking decisions.
Reads RGB directly from the original rgb/ directory (PINHOLE + zero
distortion means COLMAP's undistorted copy is pixel-identical, so we skip
regenerating dense/images, which run_mvs.py prunes after export).

Output: {workspace}/seg/{image_name}.png (uint8, one Cityscapes class id
per pixel, lossless PNG so it stays small and exact).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import fusion_config as cfg

_model = None
_processor = None


def _load_model(device: str):
    global _model, _processor
    if _model is None:
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        _processor = SegformerImageProcessor.from_pretrained(cfg.SEGFORMER_MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.SEGFORMER_MODEL_ID)
        _model.to(device).eval()
    return _model, _processor


@torch.no_grad()
def segment_batch(image_paths: list[Path], device: str) -> list[np.ndarray]:
    """Returns one (H, W) uint8 class-id array per input image, at that
    image's native resolution."""
    model, processor = _load_model(device)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    sizes = [im.size[::-1] for im in images]  # (H, W) per image, PIL gives (W,H)

    inputs = processor(images=images, return_tensors="pt").to(device)
    logits = model(**inputs).logits  # (B, 19, h, w) at model resolution

    out = []
    for i, (h, w) in enumerate(sizes):
        up = torch.nn.functional.interpolate(
            logits[i:i + 1], size=(h, w), mode="bilinear", align_corners=False)
        cls = up.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        out.append(cls)
    return out


def run_segmentation(scene, image_dir: Path, image_names: list[str],
                     device: str, batch_size: int = 4,
                     skip_existing: bool = True) -> dict:
    seg_dir = scene.workspace / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for name in image_names:
        out_path = seg_dir / f"{name}.png"
        if skip_existing and out_path.exists():
            continue
        todo.append(name)

    n_done, n_errors = 0, 0
    for i in range(0, len(todo), batch_size):
        batch_names = todo[i:i + batch_size]
        paths = [image_dir / n for n in batch_names]
        present = [(n, p) for n, p in zip(batch_names, paths) if p.exists()]
        if not present:
            n_errors += len(batch_names)
            continue
        try:
            maps = segment_batch([p for _, p in present], device)
            for (n, _), cls in zip(present, maps):
                Image.fromarray(cls, mode="L").save(seg_dir / f"{n}.png")
                n_done += 1
        except Exception as e:  # noqa: BLE001 — one bad batch never kills the scene
            print(f"[segmentation] error on batch starting {batch_names[0]}: {e}",
                  flush=True)
            n_errors += len(present)

    return {"n_requested": len(image_names), "n_computed": n_done,
            "n_skipped_existing": len(image_names) - len(todo),
            "n_errors": n_errors}
