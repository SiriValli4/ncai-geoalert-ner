#!/usr/bin/env python3
"""
train_hazard_yolov8.py

Fine-tunes a YOLOv8 object detector (via Ultralytics) to spot road-surface
hazards in field photos, for the four target classes:

    0: road_crack
    1: active_mudslide
    2: blocked_road
    3: normal            (a clear/undamaged road surface -- a genuine
                           positive detection, not just "nothing found")

--------------------------------------------------------------------------
Real training data (what you should actually use)
--------------------------------------------------------------------------
YOLO expects, under --data-dir:

    <data-dir>/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml

Each image's label file (same basename, .txt) has one line per hazard
instance:

    <class_id> <x_center> <y_center> <width> <height>

with all four numbers normalized to [0, 1] relative to image size. Tools
like Roboflow, CVAT, or LabelImg export directly to this format -- label
each field photo with a box around the crack/mudslide/blockage (or, for a
clean road, a box around the visibly normal surface) and export as
"YOLOv8"/"YOLO Darknet" format into this layout.

Then train on your real data with, e.g.:

    python train_hazard_yolov8.py --data-dir data/hazards --epochs 100 \
        --imgsz 640 --batch 16 --output hazard_yolov8.pt

--------------------------------------------------------------------------
No labeled photos yet? Smoke-test the pipeline with synthetic data
--------------------------------------------------------------------------
    python train_hazard_yolov8.py --generate-sample --data-dir sample_data \
        --epochs 15 --imgsz 320 --batch 8 --output hazard_yolov8_demo.pt

This procedurally draws simple road-like scenes with a jagged line
("crack"), a brown blob ("mudslide"), a debris-pile rectangle
("blocked_road"), or nothing ("normal"), and trains on those. It proves
the full train -> detect -> draw -> serve pipeline works end-to-end, but
the resulting weights have NOT seen a real photograph and will NOT
reliably classify real-world hazards. Swap in real, labeled field photos
before using this for anything operational.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

# See hazard_detection_api.py for why this is set before ultralytics is
# imported: it stops the library from silently pip-installing optional
# extras on its own.
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "True")

import cv2
import numpy as np
import yaml

CLASS_NAMES = ["road_crack", "active_mudslide", "blocked_road", "normal"]


# --------------------------------------------------------------------------- #
# Synthetic dataset generator (demo/smoke-test only -- see module docstring)
# --------------------------------------------------------------------------- #
def _road_background(img_size: int, rng: np.random.Generator) -> np.ndarray:
    base_gray = int(rng.integers(90, 140))
    img = np.full((img_size, img_size, 3), base_gray, dtype=np.uint8)
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # A couple of faint lane-marking-ish strokes so "normal" isn't just flat gray
    for _ in range(2):
        y = int(rng.integers(0, img_size))
        cv2.line(img, (0, y), (img_size, y + int(rng.integers(-15, 15))),
                  color=(base_gray + 15,) * 3, thickness=1)
    return img


def _draw_crack(img: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    x, y = int(rng.integers(w * 0.1, w * 0.5)), int(rng.integers(h * 0.1, h * 0.5))
    points = [(x, y)]
    for _ in range(rng.integers(4, 7)):
        x += int(rng.integers(-15, 40))
        y += int(rng.integers(-15, 40))
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        points.append((x, y))
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(img, p1, p2, color=(25, 25, 25), thickness=int(rng.integers(2, 4)))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    pad = 8
    return (max(min(xs) - pad, 0), max(min(ys) - pad, 0),
            min(max(xs) + pad, w), min(max(ys) + pad, h))


def _draw_mudslide(img: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(w * 0.3, w * 0.7)), int(rng.integers(h * 0.3, h * 0.7))
    r = int(rng.integers(int(w * 0.15), int(w * 0.3)))
    pts = []
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        radius = r * rng.uniform(0.6, 1.1)
        pts.append((int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle))))
    pts_arr = np.array(pts, dtype=np.int32)
    brown = (int(rng.integers(20, 45)), int(rng.integers(50, 80)), int(rng.integers(80, 120)))
    cv2.fillPoly(img, [pts_arr], color=brown)
    x, y, bw, bh = cv2.boundingRect(pts_arr)
    return (x, y, x + bw, y + bh)


def _draw_blocked_road(img: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    y0 = int(rng.integers(h * 0.35, h * 0.55))
    y1 = y0 + int(rng.integers(int(h * 0.15), int(h * 0.3)))
    x0 = int(rng.integers(0, w * 0.15))
    x1 = w - int(rng.integers(0, w * 0.15))
    debris_color = (int(rng.integers(60, 90)), int(rng.integers(60, 90)), int(rng.integers(60, 90)))
    cv2.rectangle(img, (x0, y0), (x1, y1), color=debris_color, thickness=-1)
    for _ in range(15):
        rx = int(rng.integers(x0, x1))
        ry = int(rng.integers(y0, y1))
        cv2.circle(img, (rx, ry), int(rng.integers(3, 8)),
                   color=(int(rng.integers(30, 60)),) * 3, thickness=-1)
    return (x0, y0, x1, y1)


def generate_sample_dataset(data_dir: Path, imgs_per_class_train: int, imgs_per_class_val: int, img_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    for split, n_per_class in (("train", imgs_per_class_train), ("val", imgs_per_class_val)):
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for class_id, class_name in enumerate(CLASS_NAMES):
            for i in range(n_per_class):
                img = _road_background(img_size, rng)
                if class_name == "road_crack":
                    box = _draw_crack(img, rng)
                elif class_name == "active_mudslide":
                    box = _draw_mudslide(img, rng)
                elif class_name == "blocked_road":
                    box = _draw_blocked_road(img, rng)
                else:  # normal -- box the visibly clear road surface itself
                    margin = int(img_size * 0.08)
                    box = (margin, margin, img_size - margin, img_size - margin)

                fname = f"{class_name}_{i:03d}"
                cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)

                x0, y0, x1, y1 = box
                xc = ((x0 + x1) / 2) / img_size
                yc = ((y0 + y1) / 2) / img_size
                bw = (x1 - x0) / img_size
                bh = (y1 - y0) / img_size
                (lbl_dir / f"{fname}.txt").write_text(
                    f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                )

    data_yaml = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    (data_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"[INFO] Synthetic dataset written to '{data_dir}' "
          f"({imgs_per_class_train} train + {imgs_per_class_val} val images per class).")
    print("[INFO] This is procedurally generated demo data, not real hazard photos.")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO  # imported lazily so --help doesn't need torch installed

    data_yaml_path = args.data_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"'{data_yaml_path}' not found. Point --data-dir at a folder with "
            f"images/, labels/, and data.yaml (see this script's docstring), "
            f"or pass --generate-sample to create a synthetic one for testing."
        )

    model = YOLO(args.weights)
    print(f"[INFO] Starting from '{args.weights}', training on '{data_yaml_path}' "
          f"for {args.epochs} epochs (imgsz={args.imgsz}, batch={args.batch}, device={args.device}) ...")

    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.runs_dir),
        name=args.run_name,
        exist_ok=True,
        verbose=False,
        patience=max(args.epochs, 20),  # don't early-stop a short smoke-test run
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        # Some very short runs only produce last.pt
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.output)
    print(f"[INFO] Trained weights copied to '{args.output}'.")

    trained_model = YOLO(str(args.output))
    print(f"[INFO] Final model classes: {trained_model.names}")
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for road hazard detection.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hazards"),
                        help="Dataset root containing images/, labels/, data.yaml.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="Starting checkpoint (pretrained COCO weights by default, "
                             "downloaded automatically on first use).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu', '0' for first GPU, etc.")
    parser.add_argument("--output", type=Path, default=Path("hazard_yolov8.pt"),
                        help="Where to copy the best trained weights.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default="hazard_train")

    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a synthetic dataset at --data-dir before training. "
                             "See module docstring -- demo/smoke-test only.")
    parser.add_argument("--sample-per-class-train", type=int, default=25)
    parser.add_argument("--sample-per-class-val", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset(
            args.data_dir, args.sample_per_class_train, args.sample_per_class_val,
            img_size=args.imgsz, seed=args.seed,
        )

    train(args)


if __name__ == "__main__":
    main()
