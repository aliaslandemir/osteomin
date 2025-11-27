#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple overlay builder for confocal stacks.

- Normalizes each channel using 1st/99th percentiles.
- Composes screen-blended RGB.
- Adds scale bar, channel legend, and ALP-style two-line condition label.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tf
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

COLOR_DAPI = np.array([0.15, 0.45, 1.00])
COLOR_ACTIN = np.array([0.92, 0.16, 0.20])
COLOR_COLLAGEN = np.array([0.12, 0.80, 0.32])

PIX_CONVERSION = 0.21
CHANNEL_SMOOTHING = {"dapi": 0.2, "actin": 0.35, "collagen": 1.1}
CHANNEL_UNSHARP = {
    "dapi": (0.45, 0.95),
    "actin": (0.55, 0.85),
    "collagen": (0.6, 0.5),
}
BACKGROUND_DROP = 0.04
CHANNEL_GAMMA = {"dapi": 1.0, "actin": 1.0, "collagen": 1.1}
CHANNEL_SCALE = {"dapi": 1.5, "actin": 1.3, "collagen": 0.55}
BACKGROUND_MUTING = 0.92

COND_MAP = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo+D/MK4 (L/L)",
    "C4": "Osteo+D/MK4 (L/H)",
    "C5": "Osteo+D/MK4 (H/L)",
    "C6": "Osteo+D/MK4 (H/H)",
    "C7": "Osteo+Dex (H)",
    "C8": "Osteo+MK-4 (H)",
}

COND_LABEL_LINES = {
    "Basal": ("Basal", ""),
    "Osteo": ("A+β-GP", ""),
    "Osteo+D/MK4 (L/L)": ("A+β-GP", "D_LK_L"),
    "Osteo+D/MK4 (L/H)": ("A+β-GP", "D_LK_H"),
    "Osteo+D/MK4 (H/L)": ("A+β-GP", "D_HK_L"),
    "Osteo+D/MK4 (H/H)": ("A+β-GP", "D_HK_H"),
    "Osteo+Dex (H)": ("A+β-GP", "D_HK_0"),
    "Osteo+MK-4 (H)": ("A+β-GP", "D_0K_H"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("input_images"),
        help="Input folder containing multichannel images",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("overlay_output"),
        help="Output directory root",
    )
    ap.add_argument("--dapi", type=int, default=0)
    ap.add_argument("--actin", type=int, default=1)
    ap.add_argument("--collagen", type=int, default=2)
    ap.add_argument("--pix", type=float, default=PIX_CONVERSION, help="µm per pixel")
    ap.add_argument("--scalebar_um", type=float, default=None)
    return ap.parse_args()


def list_images(indir: Path) -> List[Path]:
    return sorted(
        p
        for p in indir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    )


def detect_code(name: str) -> str:
    m = re.match(r"^(C[1-8])[_-]", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def condition_label_lines(code: str) -> Tuple[str, str]:
    pretty = COND_MAP.get(code, "Unknown")
    return COND_LABEL_LINES.get(pretty, (pretty, ""))


def pick_channel(arr: np.ndarray, index: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] <= 6:
            return arr[index].astype(np.float32)
        if arr.shape[-1] <= 6:
            return arr[..., index].astype(np.float32)
        return np.max(arr, axis=0).astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[1] <= 6:
            return arr[0, index].astype(np.float32)
        if arr.shape[0] <= 6:
            return arr[index, 0].astype(np.float32)
        return arr.max(axis=0)[index].astype(np.float32)
    data = arr
    while data.ndim > 2:
        data = data.max(axis=0)
    return data.astype(np.float32)


def unsharp(image: np.ndarray, radius: float, amount: float) -> np.ndarray:
    if radius <= 0 or amount <= 0:
        return image
    blur = gaussian_filter(image, sigma=radius)
    sharp = image + amount * (image - blur)
    return np.clip(sharp, 0, 1)


def prepare_channel(arr: np.ndarray, index: int, name: str) -> np.ndarray:
    scale = CHANNEL_SCALE.get(name, 1.0)
    smooth_sigma = CHANNEL_SMOOTHING.get(name, 0.4)
    channel = pick_channel(arr, index)
    if smooth_sigma > 0:
        channel = gaussian_filter(channel, sigma=smooth_sigma)
    norm = percentile_norm(channel)
    gamma = CHANNEL_GAMMA.get(name, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    radius, amount = CHANNEL_UNSHARP.get(name, (0.6, 0.7))
    if radius > 0 and amount > 0:
        norm = unsharp(norm, radius, amount)
    if BACKGROUND_DROP > 0:
        norm = np.clip(norm - BACKGROUND_DROP, 0, 1)
    if name == "collagen":
        norm = np.clip((norm - 0.05) / 0.95, 0, 1)
    norm = np.clip(norm * scale, 0, 1)
    return norm


def percentile_norm(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    lo_v = np.percentile(x, lo)
    hi_v = np.percentile(x, hi)
    if hi_v <= lo_v:
        hi_v = lo_v + 1e-6
    return np.clip((x - lo_v) / (hi_v - lo_v), 0, 1)


def screen_blend(channels: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(channels[0])
    for c in channels:
        c = np.clip(c, 0, 1)
        out = 1 - (1 - out) * (1 - c)
    return np.clip(out, 0, 1)


def compose_rgb(dapi, actin, collagen):
    r = np.clip(actin, 0, 1)[..., None] * COLOR_ACTIN
    g = np.clip(collagen, 0, 1)[..., None] * COLOR_COLLAGEN
    b = np.clip(dapi, 0, 1)[..., None] * COLOR_DAPI
    r_ch = screen_blend([r[..., 0]])
    g_ch = screen_blend([g[..., 1]])
    b_ch = screen_blend([b[..., 2]])
    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
    return np.clip(rgb * BACKGROUND_MUTING, 0, 1)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _text_width(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> int:
    if hasattr(draw, "textlength"):
        return int(draw.textlength(text, font=font))
    return font.getsize(text)[0]


def _draw_outlined_text(
    draw: ImageDraw.Draw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill=(255, 255, 255),
    outline=(0, 0, 0),
    width: float = 1.0,
):
    x, y = position
    offsets = [
        (-width, -width),
        (-width, width),
        (width, -width),
        (width, width),
        (0, -width),
        (0, width),
        (-width, 0),
        (width, 0),
    ]
    for dx, dy in offsets:
        draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def add_scale_bar_pil(im: Image.Image, pix_um: float, bar_um: float = None, color=(255, 255, 255)):
    W, H = im.size
    if bar_um is None:
        target_um = (W * pix_um) / 5.0
        nice = np.array([5, 10, 20, 25, 50, 100, 200, 500, 1000], dtype=float)
        bar_um = float(nice[np.argmin(np.abs(nice - target_um))])
    bar_px = max(int(round(bar_um / pix_um)), 1)
    pad = 16
    height_px = 8
    x0 = pad
    y0 = H - pad - height_px
    draw = ImageDraw.Draw(im)
    draw.rectangle([x0, y0, x0 + bar_px, y0 + height_px], fill=color)
    txt = f"{int(bar_um) if float(bar_um).is_integer() else bar_um} µm"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = font.getsize(txt)[0]
    draw.text((x0 + bar_px / 2 - tw / 2, y0 - 18), txt, fill=color, font=font)
    return im


def add_channel_banner(im: Image.Image) -> Image.Image:
    W, H = im.size
    banner_h = 72
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    banner = Image.new("RGBA", (W, banner_h), (0, 0, 0, 180))
    overlay.paste(banner, (0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    items = [
        ("DAPI", COLOR_DAPI),
        ("Actin", COLOR_ACTIN),
        ("Collagen I", COLOR_COLLAGEN),
    ]
    box = 24
    gap = 18
    label_widths = [_text_width(draw, label, font) for label, _ in items]
    total_w = sum(box + 10 + tw for tw in label_widths) + gap * (len(items) - 1)
    x = W - total_w - 16
    y = 12
    for (label, rgb), tw in zip(items, label_widths):
        color = tuple(int(np.clip(round(c * 255), 0, 255)) for c in rgb)
        draw.rectangle([x, y, x + box, y + box], fill=color, outline=(255, 255, 255), width=2)
        _draw_outlined_text(draw, (x + box + 10, y), label, font=font)
        x += box + 10 + tw + gap



    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def add_condition_label(im: Image.Image, label_lines: Tuple[str, str]) -> Image.Image:
    draw = ImageDraw.Draw(im)
    try:
        base_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        base_font = ImageFont.load_default()
    try:
        italic_font = ImageFont.truetype("DejaVuSans-Oblique.ttf", 20)
    except Exception:
        italic_font = base_font
    pad = 16
    y = pad
    line1, line2 = label_lines
    if line1:
        _draw_outlined_text(draw, (pad, y), line1, base_font, width=1.5)
        y += base_font.getbbox(line1)[3] - base_font.getbbox(line1)[1] + 2 if hasattr(base_font, "getbbox") else base_font.getsize(line1)[1] + 2
    if line2:
        _draw_outlined_text(draw, (pad, y), line2, italic_font, width=1.0)
    return im


def main():
    args = parse_args()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "overlays").mkdir(exist_ok=True)

    paths = list_images(args.input)
    if not paths:
        print(f"[!] No images found in {args.input}")
        return

    for path in paths:
        arr = tf.imread(path)
        dapi = prepare_channel(arr, args.dapi, "dapi")
        actin = prepare_channel(arr, args.actin, "actin")
        collagen = prepare_channel(arr, args.collagen, "collagen")

        rgb = compose_rgb(dapi, actin, collagen)
        img = Image.fromarray(to_uint8(rgb))
        img = add_channel_banner(img)
        img = add_scale_bar_pil(img, args.pix, bar_um=args.scalebar_um)
        code = detect_code(path.name)
        label = condition_label_lines(code)
        img = add_condition_label(img, label)

        out_path = out_root / "overlays" / f"{path.stem}_overlay.png"
        img.save(out_path, dpi=(100, 100))
        print(f"Saved overlay: {out_path}")


if __name__ == "__main__":
    main()
