#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

HOURS_PER_DAY = 24
TOTAL_HOURS_DEFAULT = 672
MODALITIES = ("spatial_cells", "alp", "collagen_ha")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run OsteoMin baseline settings and export hourly PNG frames into the same "
            "day_hor/day*/(spatial_cells|alp|collagen_ha) folders."
        )
    )
    p.add_argument("--out", type=Path, default=Path("main_out"), help="Output directory")
    p.add_argument("--dpi", type=int, default=300, help="PNG rasterization DPI")
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run main.py",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip simulation and process existing day_hor PDFs",
    )
    p.add_argument(
        "--no-annotate-hour",
        action="store_true",
        help="Do not add cumulative hour label overlay to PNGs",
    )
    p.add_argument(
        "--remove-hourly-pdf",
        action="store_true",
        help="Delete hourly PDFs after PNG conversion",
    )
    return p.parse_args()


def build_main_cmd(python_bin: str, outdir: Path) -> list[str]:
    return [
        python_bin,
        "main.py",
        "--out",
        str(outdir),
        "--seed",
        "12345",
        "--runtime",
        "--hours",
        "672",
        "--cells",
        "72",
        "--w",
        "33",
        "--h",
        "33",
        "--boundary-mode",
        "periodic",
        "--seed-mode",
        "random",
        "--media-interval",
        "48",
        "--secretion",
        "--aa0",
        "50.0",
        "--aa-half-life",
        "8",
        "--collagen-aa-km",
        "5.0",
        "--random-move",
        "--move-prob",
        "0.27",
        "--max-moves",
        "1",
        "--proliferation",
        "--ob-div-prob",
        "0.0",
        "--min-div-hours",
        "38",
        "--max-div-hours",
        "52",
        "--apoptosis",
        "--apoptosis-prob",
        "0.0010",
        "--apoptosis-osteo",
        "0.0",
        "--osteogenic-commitment",
        "--osteo-induction-hours",
        "72",
        "--commit-delay",
        "48",
        "--commit-k",
        "0.0125",
        "--dex-conc",
        "0",
        "--mk4-conc",
        "0",
        "--alp-rb",
        "0.0949",
        "--alp-ro",
        "0.810",
        "--collagen-rb",
        "0.00336",
        "--collagen-ro",
        "0.0130",
        "--ha-ro",
        "0.102",
        "--alp-dex-ref",
        "22.7",
        "--collagen-dex-ref",
        "6.22",
        "--ars-dex-ref",
        "46.6",
        "--alp-a-d",
        "3.72",
        "--alp-b-d",
        "1.75",
        "--collagen-a-d",
        "2.69",
        "--collagen-b-d",
        "2.04",
        "--ha-a-d",
        "1.22",
        "--ha-b-d",
        "0.679",
        "--alp-mk4-ref",
        "0.784",
        "--collagen-mk4-ref",
        "0.349",
        "--ars-mk4-ref",
        "0.667",
        "--alp-a-k",
        "1.10",
        "--alp-b-k",
        "1.06",
        "--collagen-a-k",
        "4.66",
        "--collagen-b-k",
        "2.15",
        "--ha-a-k",
        "4.66",
        "--ha-b-k",
        "2.15",
        "--ha-model",
        "ha",
        "--ha-collagen-maturation-tau",
        "48",
        "--ha-collagen-km",
        "1.0",
        "--ha-tau",
        "384",
        "--ha-hill-n",
        "2",
        "--spatial-every",
        "24",
        "--alp-plot-hours",
        "240",
        "--collagen-plot-hours",
        "336",
        "--mineral-plot-hours",
        "672",
        "--hourly-exports",
        "--alp-snapshots",
    ]


def ensure_tool(name: str, install_hint: str) -> None:
    from shutil import which

    if which(name) is None:
        raise SystemExit(f"{name} not found. {install_hint}")


def collect_hourly_pdfs(outdir: Path) -> dict[str, list[Path]]:
    root = outdir / "day_hor"
    result: dict[str, list[Path]] = {}
    for modality in MODALITIES:
        result[modality] = sorted(root.glob(f"day*/{modality}/hour_*.pdf"))
    return result


def parse_day_hour(path: Path) -> tuple[int, int]:
    day_match = re.search(r"day(\d+)", path.as_posix())
    hour_match = re.search(r"hour_(\d+)\.(?:pdf|png)$", path.name)
    if not day_match or not hour_match:
        return (-1, -1)
    return int(day_match.group(1)), int(hour_match.group(1))


def absolute_hour(day: int, hour_in_day: int) -> int:
    return (day - 1) * HOURS_PER_DAY + hour_in_day


def convert_pdf_to_png(pdf_path: Path, dpi: int) -> Path:
    prefix = pdf_path.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            str(dpi),
            str(pdf_path),
            str(prefix),
        ],
        check=True,
    )
    return prefix.with_suffix(".png")


def annotate_png_hour(png_path: Path, hour_abs: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(png_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, img.width // 48))
    except Exception:
        font = ImageFont.load_default()

    text = f"Hour {hour_abs:03d} / {TOTAL_HOURS_DEFAULT - 1:03d}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    px = max(8, img.width // 60)
    py = max(6, img.height // 90)
    x = (img.width - tw) // 2
    y = max(8, img.height // 70)

    draw.rectangle(
        [x - px, y - py, x + tw + px, y + th + py],
        fill=(255, 255, 255, 220),
        outline=(0, 0, 0, 160),
        width=2,
    )
    draw.text((x, y), text, fill=(15, 23, 42, 255), font=font)

    img.convert("RGB").save(png_path)


def write_manifest(outdir: Path, png_by_modality: dict[str, list[Path]]) -> Path:
    manifest = outdir / "hourly_manifest_png.csv"
    by_hour: dict[str, dict[int, Path]] = {}
    all_hours: set[int] = set()
    for modality, paths in png_by_modality.items():
        mod_map: dict[int, Path] = {}
        for p in paths:
            day, hour = parse_day_hour(p)
            if day < 1 or hour < 0:
                continue
            h_abs = absolute_hour(day, hour)
            mod_map[h_abs] = p
            all_hours.add(h_abs)
        by_hour[modality] = mod_map

    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hour_abs", "day", "hour_in_day", *MODALITIES])
        for h_abs in sorted(all_hours):
            day = h_abs // HOURS_PER_DAY + 1
            h_day = h_abs % HOURS_PER_DAY
            w.writerow(
                [
                    h_abs,
                    day,
                    h_day,
                    by_hour["spatial_cells"].get(h_abs, ""),
                    by_hour["alp"].get(h_abs, ""),
                    by_hour["collagen_ha"].get(h_abs, ""),
                ]
            )
    return manifest


def main() -> None:
    args = parse_args()
    outdir = args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        cmd = build_main_cmd(args.python, outdir)
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_osteomin")
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    ensure_tool("pdftoppm", "Install poppler-utils, e.g. `sudo apt install poppler-utils`.")

    if not args.no_annotate_hour:
        ensure_tool("python", "A Python interpreter with Pillow is required.")

    pdf_frames = collect_hourly_pdfs(outdir)
    png_frames: dict[str, list[Path]] = {k: [] for k in MODALITIES}

    for modality in MODALITIES:
        paths = pdf_frames[modality]
        print(f"[pdf] {modality}: {len(paths)}")
        for pdf in paths:
            png = convert_pdf_to_png(pdf, args.dpi)
            day, hour = parse_day_hour(png)
            if day >= 1 and hour >= 0 and not args.no_annotate_hour:
                annotate_png_hour(png, absolute_hour(day, hour))
            png_frames[modality].append(png)
        print(f"[png] {modality}: {len(png_frames[modality])}")

    manifest = write_manifest(outdir, png_frames)
    print(f"[manifest] {manifest}")

    if args.remove_hourly_pdf:
        removed = 0
        for modality in MODALITIES:
            for pdf in pdf_frames[modality]:
                if pdf.exists():
                    pdf.unlink()
                    removed += 1
        print(f"[cleanup] removed hourly pdf files: {removed}")

    print("[done] PNG hourly exports ready")


if __name__ == "__main__":
    main()
