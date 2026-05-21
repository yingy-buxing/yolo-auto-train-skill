#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ROOT = Path(os.environ.get("YOLO_AUTO_TRAIN_ROOT", Path.cwd() / "yolo-auto-train-runs"))
MODE_PRESETS = {
    "fast": {
        "sample_fps": 0.5,
        "max_frames": 300,
        "conf": 0.20,
        "imgsz": 640,
        "sam_model": "none",
        "epochs": 25,
        "min_positive_frames": 20,
        "predict_samples": 16,
    },
    "balanced": {
        "sample_fps": 1.0,
        "max_frames": 1200,
        "conf": 0.10,
        "imgsz": 640,
        "sam_model": "mobile_sam.pt",
        "epochs": 50,
        "min_positive_frames": 50,
        "predict_samples": 24,
    },
    "quality": {
        "sample_fps": 2.0,
        "max_frames": 3000,
        "conf": 0.05,
        "imgsz": 960,
        "sam_model": "mobile_sam.pt",
        "epochs": 100,
        "min_positive_frames": 120,
        "predict_samples": 40,
    },
}


@dataclass
class Box:
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass
class SamStats:
    refined: int = 0
    fallback: int = 0
    unavailable: int = 0


def log(message: str) -> None:
    print(f"[yolo-auto-train] {message}", flush=True)


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = MODE_PRESETS[args.mode]
    applied: list[str] = []
    for key, value in preset.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
            applied.append(f"{key}={value}")
    if applied:
        log(f"mode={args.mode} defaults: {', '.join(applied)}")
    return args


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if slug:
        return slug[:48]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"target-{digest}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_workdir(args: argparse.Namespace) -> Path:
    if getattr(args, "workdir", None):
        return Path(args.workdir).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (DEFAULT_ROOT / "runs" / f"{slugify(args.target)}-{stamp}").resolve()


def scan_videos(raw: str | None) -> list[Path]:
    if not raw:
        default = DEFAULT_ROOT / "videos"
        return sorted(p for p in default.rglob("*") if p.suffix.lower() in VIDEO_EXTS) if default.exists() else []
    path = Path(raw).resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTS else []
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return []


def scan_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def average_hash(frame: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_frames(args: argparse.Namespace) -> Path:
    workdir = resolve_workdir(args)
    frames_dir = ensure_dir(workdir / "frames")
    reports_dir = ensure_dir(workdir / "reports")
    videos = scan_videos(args.videos)
    if not videos:
        raise SystemExit(f"No videos found. Put videos in {DEFAULT_ROOT / 'videos'} or pass --videos.")

    log(f"extracting frames from {len(videos)} video(s) into {frames_dir}")
    saved = 0
    seen_hashes: list[int] = []
    metadata_rows: list[dict[str, str | int | float]] = []

    for video_index, video in enumerate(videos):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            log(f"warning: could not open {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / max(args.sample_fps, 0.001))))
        frame_index = 0
        while saved < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            if args.dedupe:
                ahash = average_hash(frame)
                if any(hamming(ahash, old) <= args.dedupe_hamming for old in seen_hashes[-300:]):
                    frame_index += 1
                    continue
                seen_hashes.append(ahash)
            name = f"v{video_index:03d}_f{frame_index:08d}.jpg"
            out = frames_dir / name
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            metadata_rows.append(
                {
                    "image": name,
                    "video": str(video),
                    "frame_index": frame_index,
                    "time_sec": round(frame_index / fps, 3),
                }
            )
            saved += 1
            if saved % 100 == 0:
                log(f"saved {saved} frames")
            frame_index += 1
        cap.release()
        if saved >= args.max_frames:
            break

    write_csv(workdir / "frames.csv", metadata_rows, ["image", "video", "frame_index", "time_sec"])
    make_contact_sheet(scan_images(frames_dir), reports_dir / "frame_sheet.jpg", title=f"Extracted frames: {saved}")
    write_report(workdir, extra={"stage": "extract", "frames": saved, "videos": len(videos)})
    return workdir


def frame_quality_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    brightness_penalty = abs(brightness - 115.0) * 2.0
    return blur - brightness_penalty


def extract_prompt_samples(args: argparse.Namespace) -> Path:
    workdir = resolve_workdir(args)
    prompt_dir = workdir / "prompt_frames"
    reports_dir = ensure_dir(workdir / "reports")
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir)
    ensure_dir(prompt_dir)
    videos = scan_videos(args.videos)
    if not videos:
        raise SystemExit(f"No videos found. Put videos in {DEFAULT_ROOT / 'videos'} or pass --videos.")

    target_count = max(1, args.prompt_samples)
    candidates: list[tuple[float, int, int, float, Path, np.ndarray]] = []
    for video_index, video in enumerate(videos):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            log(f"warning: could not open {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, total // max(target_count * 8, 1)) if total else max(1, int(round(fps)))
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                score = frame_quality_score(frame)
                candidates.append((score, video_index, frame_index, frame_index / fps, video, frame.copy()))
            frame_index += 1
        cap.release()

    if not candidates:
        raise SystemExit("No prompt sample frames could be extracted.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = sorted(candidates[:target_count], key=lambda item: (item[1], item[2]))
    rows: list[dict[str, str | int | float]] = []
    for out_index, (score, video_index, frame_index, time_sec, video, frame) in enumerate(chosen):
        name = f"p{out_index:03d}_v{video_index:03d}_f{frame_index:08d}.jpg"
        out = prompt_dir / name
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        rows.append(
            {
                "image": name,
                "video": str(video),
                "frame_index": frame_index,
                "time_sec": round(time_sec, 3),
                "quality_score": round(score, 3),
            }
        )

    write_csv(workdir / "prompt_frames.csv", rows, ["image", "video", "frame_index", "time_sec", "quality_score"])
    make_contact_sheet(scan_images(prompt_dir), reports_dir / "prompt_sheet.jpg", title=f"Prompt discovery frames: {len(rows)}", max_items=target_count)
    write_report(
        workdir,
        extra={
            "stage": "prompt-discovery",
            "status": "needs_ai_prompt",
            "prompt_samples": len(rows),
            "prompt_sheet": str(reports_dir / "prompt_sheet.jpg"),
        },
    )
    return workdir


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evenly_spaced(items: list[Path], max_items: int) -> list[Path]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return [items[0]]
    indexes = [round(i * (len(items) - 1) / (max_items - 1)) for i in range(max_items)]
    return [items[i] for i in indexes]


def result_boxes(result, conf_floor: float) -> list[Box]:
    boxes: list[Box] = []
    raw = getattr(result, "boxes", None)
    if raw is None or len(raw) == 0:
        return boxes
    xyxy = raw.xyxy.detach().cpu().numpy()
    confs = raw.conf.detach().cpu().numpy() if raw.conf is not None else np.ones((len(xyxy),), dtype=float)
    for coords, conf in zip(xyxy, confs):
        if float(conf) >= conf_floor:
            x1, y1, x2, y2 = [float(v) for v in coords]
            boxes.append(Box(cls=0, conf=float(conf), xyxy=(x1, y1, x2, y2)))
    return boxes


def mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def maybe_refine_with_sam(image: Path, boxes: list[Box], sam_model, device: str, imgsz: int, stats: SamStats | None = None) -> list[Box]:
    if not boxes or sam_model is None:
        if boxes and stats is not None:
            stats.unavailable += len(boxes)
        return boxes
    try:
        bboxes = np.array([b.xyxy for b in boxes], dtype=np.float32)
        results = sam_model.predict(source=str(image), bboxes=bboxes, device=device, imgsz=imgsz, verbose=False)
    except Exception as exc:
        log(f"warning: SAM refinement failed for {image.name}: {exc}")
        if stats is not None:
            stats.unavailable += len(boxes)
        return boxes
    if not results:
        if stats is not None:
            stats.unavailable += len(boxes)
        return boxes
    masks = getattr(results[0], "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        if stats is not None:
            stats.unavailable += len(boxes)
        return boxes
    data = masks.data.detach().cpu().numpy()
    refined: list[Box] = []
    for original, mask in zip(boxes, data):
        new_box = mask_to_box(mask)
        if new_box is not None and is_reasonable_refinement(original.xyxy, new_box):
            refined.append(Box(cls=0, conf=original.conf, xyxy=new_box))
            if stats is not None:
                stats.refined += 1
        else:
            refined.append(original)
            if stats is not None:
                stats.fallback += 1
    if stats is not None and len(data) < len(boxes):
        stats.unavailable += len(boxes) - len(data)
        refined.extend(boxes[len(data) :])
    return refined or boxes


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_reasonable_refinement(original: tuple[float, float, float, float], refined: tuple[float, float, float, float]) -> bool:
    original_area = box_area(original)
    refined_area = box_area(refined)
    if original_area <= 1 or refined_area <= 1:
        return False
    area_ratio = refined_area / original_area
    if area_ratio < 0.30 or area_ratio > 1.20:
        return False
    ox, oy = box_center(original)
    rx, ry = box_center(refined)
    x1, y1, x2, y2 = original
    diag = math.hypot(x2 - x1, y2 - y1)
    if diag <= 1:
        return False
    center_shift = math.hypot(rx - ox, ry - oy) / diag
    return center_shift <= 0.25


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = box_area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    union = box_area(left) + box_area(right) - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicate_boxes(boxes: list[Box], iou_threshold: float) -> tuple[list[Box], int]:
    if len(boxes) <= 1 or iou_threshold <= 0:
        return boxes, 0
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.conf, reverse=True):
        if any(box_iou(box.xyxy, existing.xyxy) >= iou_threshold for existing in kept):
            continue
        kept.append(box)
    return kept, len(boxes) - len(kept)


def load_open_vocab_model(args: argparse.Namespace):
    from ultralytics import YOLOE, YOLOWorld

    errors: list[str] = []
    candidates: list[tuple[str, str]]
    if args.label_backend == "yoloe":
        candidates = [("yoloe", args.label_model or "yoloe-26n-seg.pt"), ("world", "yolov8s-worldv2.pt")]
    elif args.label_backend == "world":
        candidates = [("world", args.label_model or "yolov8s-worldv2.pt"), ("yoloe", "yoloe-26n-seg.pt")]
    else:
        candidates = [("world", args.label_model or "yolov8s-worldv2.pt")]

    for kind, model_name in candidates:
        try:
            model = YOLOE(model_name) if kind == "yoloe" else YOLOWorld(model_name)
            model.set_classes([args.prompt or args.target])
            log(f"loaded {kind} auto-label model: {model_name}")
            return model, kind, model_name
        except Exception as exc:
            errors.append(f"{kind}:{model_name}: {exc}")
    raise SystemExit("Could not load an open-vocabulary label model:\n" + "\n".join(errors))


def load_sam_model(args: argparse.Namespace):
    if args.sam_model.lower() in {"none", "off", "false", "0"}:
        return None
    try:
        from ultralytics import SAM

        model = SAM(args.sam_model)
        log(f"loaded SAM refinement model: {args.sam_model}")
        return model
    except Exception as exc:
        log(f"warning: SAM model '{args.sam_model}' unavailable; continuing without refinement: {exc}")
        return None


def xyxy_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, 0.0), float(width - 1))
    y2 = min(max(y2, 0.0), float(height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, bw / width, bh / height)


def write_label(path: Path, boxes: list[Box], image_path: Path) -> int:
    image = Image.open(image_path)
    width, height = image.size
    lines: list[str] = []
    for box in boxes:
        converted = xyxy_to_yolo(box.xyxy, width, height)
        if converted is None:
            continue
        xc, yc, bw, bh = converted
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def label_frames(args: argparse.Namespace) -> Path:
    workdir = resolve_workdir(args)
    frames_dir = workdir / "frames"
    labels_dir = ensure_dir(workdir / "labels")
    reports_dir = ensure_dir(workdir / "reports")
    images = scan_images(frames_dir)
    if not images:
        raise SystemExit(f"No frames found in {frames_dir}. Run extract first.")

    model, backend_kind, backend_model = load_open_vocab_model(args)
    sam_model = load_sam_model(args) if args.mode in {"balanced", "quality"} else None
    sam_stats = SamStats()
    suppressed_boxes = 0
    stats_rows: list[dict[str, str | int | float]] = []

    log(f"auto-labeling {len(images)} frame(s) for target: {args.target}")
    for idx, image in enumerate(images, 1):
        results = model.predict(source=str(image), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        boxes = result_boxes(results[0], args.conf) if results else []
        boxes = maybe_refine_with_sam(image, boxes, sam_model, args.device, args.imgsz, sam_stats)
        boxes, suppressed = suppress_duplicate_boxes(boxes, args.box_iou)
        suppressed_boxes += suppressed
        count = write_label(labels_dir / f"{image.stem}.txt", boxes, image)
        stats_rows.append({"image": image.name, "boxes": count, "max_conf": round(max([b.conf for b in boxes], default=0.0), 4)})
        if idx % 100 == 0:
            log(f"labeled {idx}/{len(images)} frames")

    write_csv(workdir / "labels.csv", stats_rows, ["image", "boxes", "max_conf"])
    make_contact_sheet(images, reports_dir / "label_sheet.jpg", title=f"Auto labels: {args.prompt or args.target}", labels_dir=labels_dir)
    prepare_dataset(args, workdir)
    positives = sum(1 for row in stats_rows if int(row["boxes"]) > 0)
    write_report(
        workdir,
        extra={
            "stage": "label",
            "frames": len(images),
            "positive_frames": positives,
            "backend": backend_kind,
            "backend_model": backend_model,
            "sam_model": args.sam_model if sam_model else "none",
            "sam_refined_boxes": sam_stats.refined,
            "sam_fallback_boxes": sam_stats.fallback,
            "sam_unavailable_boxes": sam_stats.unavailable,
            "suppressed_duplicate_boxes": suppressed_boxes,
        },
    )
    return workdir


def mark_empty(args: argparse.Namespace) -> Path:
    workdir = resolve_workdir(args)
    labels_dir = ensure_dir(workdir / "labels")
    frames = scan_images(workdir / "frames")
    if not frames:
        raise SystemExit(f"No frames found in {workdir / 'frames'}.")

    selected = select_frames_for_empty_marking(frames, args)
    if not selected:
        raise SystemExit("No frames matched the mark-empty selectors.")

    log(f"marking {len(selected)} frame(s) as empty/negative labels")
    for image in selected:
        if args.dry_run:
            log(f"would clear {image.name}")
            continue
        (labels_dir / f"{image.stem}.txt").write_text("", encoding="utf-8")

    if not args.dry_run:
        rewrite_label_stats(workdir)
        prepare_dataset(args, workdir)
        make_contact_sheet(frames, workdir / "reports" / "label_sheet.jpg", title="Curated labels", labels_dir=labels_dir)
        write_report(workdir, extra={"stage": "mark-empty", "cleared_frames": len(selected)})
    return workdir


def prune_boxes(args: argparse.Namespace) -> Path:
    workdir = resolve_workdir(args)
    labels_dir = ensure_dir(workdir / "labels")
    frames = scan_images(workdir / "frames")
    if not frames:
        raise SystemExit(f"No frames found in {workdir / 'frames'}.")

    selected = select_frames_for_empty_marking(frames, args) if (args.images or args.stems or args.first_n or args.frame_range) else frames
    removed = 0
    touched = 0
    for image in selected:
        label_path = labels_dir / f"{image.stem}.txt"
        if not label_path.exists():
            continue
        kept: list[str] = []
        frame_removed = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if yolo_line_matches_prune(line, args):
                frame_removed += 1
            else:
                kept.append(line)
        if frame_removed:
            touched += 1
            removed += frame_removed
            if args.dry_run:
                log(f"would remove {frame_removed} box(es) from {image.name}")
            else:
                label_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    if removed == 0:
        log("no boxes matched the prune filters")
    elif not args.dry_run:
        rewrite_label_stats(workdir)
        prepare_dataset(args, workdir)
        make_contact_sheet(frames, workdir / "reports" / "label_sheet.jpg", title="Pruned labels", labels_dir=labels_dir)
        write_report(workdir, extra={"stage": "prune-boxes", "pruned_boxes": removed, "touched_frames": touched})
    return workdir


def yolo_line_matches_prune(line: str, args: argparse.Namespace) -> bool:
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    _, xc, yc, bw, bh = parts
    values = {
        "xc": float(xc),
        "yc": float(yc),
        "width": float(bw),
        "height": float(bh),
        "area": float(bw) * float(bh),
    }
    checks = [
        ("min_xc", "xc", ">="),
        ("max_xc", "xc", "<="),
        ("min_yc", "yc", ">="),
        ("max_yc", "yc", "<="),
        ("min_width", "width", ">="),
        ("max_width", "width", "<="),
        ("min_height", "height", ">="),
        ("max_height", "height", "<="),
        ("min_area", "area", ">="),
        ("max_area", "area", "<="),
    ]
    active = False
    for arg_name, value_name, op in checks:
        threshold = getattr(args, arg_name)
        if threshold is None:
            continue
        active = True
        value = values[value_name]
        if op == ">=" and value < threshold:
            return False
        if op == "<=" and value > threshold:
            return False
    return active


def assist_workflow(args: argparse.Namespace) -> Path:
    workdir = resolve_workdir(args)
    args.workdir = str(workdir)
    if not args.prompt and not args.skip_label:
        if args.skip_extract:
            raise SystemExit("--skip-extract with assist still needs --prompt; inspect prompt_sheet/frame_sheet and rerun with a specific prompt.")
        extract_prompt_samples(args)
        log("assist paused after lightweight prompt discovery because no --prompt was provided")
        log("inspect reports/prompt_sheet.jpg, write a specific English prompt, then rerun assist with --prompt")
        return workdir

    if not args.skip_extract:
        extract_frames(args)
    elif not scan_images(workdir / "frames"):
        raise SystemExit(f"--skip-extract was used, but no frames were found in {workdir / 'frames'}.")

    if not args.skip_label:
        label_frames(args)
    elif not sorted((workdir / "labels").glob("*.txt")):
        raise SystemExit(f"--skip-label was used, but no labels were found in {workdir / 'labels'}.")

    if args.auto_train:
        train_model(args)
        evaluate_model(args)
        write_report(workdir, extra={"stage": "assist", "status": "auto_train_complete"})
    else:
        write_report(workdir, extra={"stage": "assist", "status": "ready_for_visual_qa"})
        log("assist paused before training; inspect reports/frame_sheet.jpg and reports/label_sheet.jpg")
        log("if labels look good, run train; if not, adjust prompt/conf or use mark-empty")
    return workdir


def select_frames_for_empty_marking(frames: list[Path], args: argparse.Namespace) -> list[Path]:
    by_name = {p.name: p for p in frames}
    by_stem = {p.stem: p for p in frames}
    selected: dict[str, Path] = {}

    for raw in args.images or []:
        for token in split_csv_tokens(raw):
            match = by_name.get(token) or by_stem.get(Path(token).stem)
            if match:
                selected[match.name] = match

    for raw in args.stems or []:
        for token in split_csv_tokens(raw):
            match = by_stem.get(token)
            if match:
                selected[match.name] = match

    if args.first_n:
        for image in frames[: args.first_n]:
            selected[image.name] = image

    for raw_range in args.frame_range or []:
        start, end = parse_frame_range(raw_range)
        for image in frames:
            frame_index = frame_index_from_name(image.name)
            if frame_index is not None and start <= frame_index <= end:
                selected[image.name] = image

    return [selected[name] for name in sorted(selected)]


def split_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def parse_frame_range(raw: str) -> tuple[int, int]:
    if ":" in raw:
        left, right = raw.split(":", 1)
    elif "-" in raw:
        left, right = raw.split("-", 1)
    else:
        value = int(raw)
        return value, value
    start, end = int(left), int(right)
    if end < start:
        start, end = end, start
    return start, end


def frame_index_from_name(name: str) -> int | None:
    match = re.search(r"_f(\d+)", name)
    return int(match.group(1)) if match else None


def rewrite_label_stats(workdir: Path) -> None:
    frames = scan_images(workdir / "frames")
    labels_dir = workdir / "labels"
    rows: list[dict[str, str | int | float]] = []
    for image in frames:
        label = labels_dir / f"{image.stem}.txt"
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()] if label.exists() else []
        rows.append({"image": image.name, "boxes": len(lines), "max_conf": ""})
    write_csv(workdir / "labels.csv", rows, ["image", "boxes", "max_conf"])


def prepare_dataset(args: argparse.Namespace, workdir: Path) -> Path:
    frames_dir = workdir / "frames"
    labels_dir = workdir / "labels"
    dataset = workdir / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    train_img = ensure_dir(dataset / "images" / "train")
    val_img = ensure_dir(dataset / "images" / "val")
    train_lbl = ensure_dir(dataset / "labels" / "train")
    val_lbl = ensure_dir(dataset / "labels" / "val")

    images = scan_images(frames_dir)
    rng = random.Random(args.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(math.ceil(len(shuffled) * args.val_ratio))) if len(shuffled) > 1 else 0
    val_set = {p.name for p in shuffled[:val_count]}

    for image in images:
        is_val = image.name in val_set
        image_dest = val_img if is_val else train_img
        label_dest = val_lbl if is_val else train_lbl
        shutil.copy2(image, image_dest / image.name)
        src_label = labels_dir / f"{image.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, label_dest / src_label.name)
        else:
            (label_dest / f"{image.stem}.txt").write_text("", encoding="utf-8")

    data = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "names": {0: args.target},
    }
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log(f"wrote dataset: {data_yaml}")
    return data_yaml


def read_yolo_boxes(label_path: Path, image_size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    width, height = image_size
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = map(float, parts)
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def make_contact_sheet(images: list[Path], out_path: Path, title: str, labels_dir: Path | None = None, max_items: int = 40) -> None:
    ensure_dir(out_path.parent)
    if not images:
        Image.new("RGB", (900, 240), "white").save(out_path)
        return
    sample = evenly_spaced(images, max_items)
    thumb_w, thumb_h = 240, 160
    cols = 5
    rows = math.ceil(len(sample) / cols)
    title_h = 42
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill=(0, 0, 0))

    for idx, image_path in enumerate(sample):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image.thumbnail((thumb_w, thumb_h))
        tile = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
        xoff = (thumb_w - image.width) // 2
        yoff = (thumb_h - image.height) // 2
        tile.paste(image, (xoff, yoff))
        tile_draw = ImageDraw.Draw(tile)
        if labels_dir is not None:
            for x1, y1, x2, y2 in read_yolo_boxes(labels_dir / f"{image_path.stem}.txt", original_size):
                sx = image.width / original_size[0]
                sy = image.height / original_size[1]
                tile_draw.rectangle((xoff + x1 * sx, yoff + y1 * sy, xoff + x2 * sx, yoff + y2 * sy), outline=(0, 220, 80), width=3)
        tile_draw.text((6, thumb_h - 18), image_path.name[:32], fill=(0, 0, 0))
        col = idx % cols
        row = idx // cols
        sheet.paste(tile, (col * thumb_w, title_h + row * thumb_h))

    sheet.save(out_path, quality=92)
    log(f"wrote preview: {out_path}")


def train_model(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    workdir = resolve_workdir(args)
    data_yaml = workdir / "dataset" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Dataset config not found: {data_yaml}. Run label first.")
    positive, boxes = label_counts(workdir)
    if positive == 0 and not getattr(args, "allow_empty_training", False):
        raise SystemExit(
            "Auto-labeling produced zero positive frames. Inspect reports/label_sheet.jpg, "
            "adjust --prompt/--conf/--label-backend, or pass --allow-empty-training for a pipeline smoke test."
        )
    if positive < getattr(args, "min_positive_frames", 10):
        log(f"warning: only {positive} positive frame(s) and {boxes} box(es); training quality will likely be poor")
    train_project = ensure_dir(workdir / "train")
    model_name = args.train_model
    log(f"training {model_name} with data {data_yaml}")
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        project=str(train_project),
        name="yolo",
        exist_ok=True,
    )
    best = train_project / "yolo" / "weights" / "best.pt"
    write_report(workdir, extra={"stage": "train", "train_model": model_name, "best": str(best), "results": str(results)})
    return workdir


def latest_best_weight(workdir: Path) -> Path | None:
    candidates = sorted(workdir.glob("train/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def evaluate_model(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    workdir = resolve_workdir(args)
    raw_weights = getattr(args, "weights", None)
    weight = Path(raw_weights).resolve() if raw_weights else latest_best_weight(workdir)
    if weight is None or not weight.exists():
        raise SystemExit("No trained best.pt found. Run train first or pass --weights.")
    images = evenly_spaced(scan_images(workdir / "frames"), args.predict_samples)
    if not images:
        raise SystemExit("No frames available for prediction samples.")
    out_dir = ensure_dir(workdir / "reports" / "predictions")
    log(f"running sample predictions with {weight}")
    model = YOLO(str(weight))
    model.predict(source=[str(p) for p in images], imgsz=args.imgsz, conf=args.conf, device=args.device, save=True, project=str(out_dir), name="samples", exist_ok=True)
    write_report(workdir, extra={"stage": "evaluate", "weights": str(weight), "prediction_dir": str(out_dir / "samples")})
    return workdir


def write_report(workdir: Path, extra: dict | None = None) -> None:
    reports_dir = ensure_dir(workdir / "reports")
    frames = scan_images(workdir / "frames")
    labels = sorted((workdir / "labels").glob("*.txt")) if (workdir / "labels").exists() else []
    positive, boxes = label_counts(workdir)
    empty = len(labels) - positive
    warnings: list[str] = []
    if labels and positive < 50:
        warnings.append("Fewer than 50 positive labeled frames; expect weak training unless the object is very simple.")
    if labels and positive / max(len(labels), 1) < 0.25:
        warnings.append("Most frames have empty labels; check whether auto-labeling missed the target.")
    if labels and boxes / max(positive, 1) > 5:
        warnings.append("Many boxes per positive frame; check for false positives or duplicate detections.")
    if labels and empty == 0:
        warnings.append("No negative/background frames; add empty labels for lookalikes and target-free frames to reduce false positives.")

    lines = [
        "# YOLO Auto Train Report",
        "",
        f"- Workdir: `{workdir}`",
        f"- Frames: {len(frames)}",
        f"- Label files: {len(labels)}",
        f"- Positive frames: {positive}",
        f"- Empty/background frames: {empty}",
        f"- Boxes: {boxes}",
    ]
    if extra:
        lines.append("")
        lines.append("## Last Stage")
        for key, value in extra.items():
            lines.append(f"- {key}: `{value}`")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
    recommendations = recommended_next_steps(workdir, extra=extra)
    if recommendations:
        lines.append("")
        lines.append("## Recommended Next Steps")
        lines.extend(f"- {item}" for item in recommendations)
    lines.extend(
        [
            "",
            "## Visual QA",
            "- Inspect `reports/frame_sheet.jpg` after extraction.",
            "- Inspect `reports/label_sheet.jpg` before training.",
            "- Use `mark-empty` on false-positive lookalike frames before training.",
            "- Inspect `reports/predictions/` after training.",
        ]
    )
    (reports_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def recommended_next_steps(workdir: Path, extra: dict | None = None) -> list[str]:
    frames = scan_images(workdir / "frames")
    labels_dir = workdir / "labels"
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    positive, _ = label_counts(workdir)
    empty = len(labels) - positive
    best = latest_best_weight(workdir)
    prediction_dir = workdir / "reports" / "predictions" / "samples"
    stage = str(extra.get("stage", "")) if extra else ""
    script = Path(__file__).resolve()

    if stage == "prompt-discovery":
        return [
            "Inspect `reports/prompt_sheet.jpg` and write a specific English prompt from the visible target and distractors.",
            f"Continue full extraction and labeling: `python {script} assist --target \"<target>\" --videos \"<video-or-folder>\" --workdir \"{workdir}\" --prompt \"<specific English prompt>\" --mode balanced`.",
        ]

    if not frames:
        return [
            f"Extract frames: `python {script} extract --target \"<target>\" --videos \"<video-or-folder>\" --mode balanced`."
        ]
    if not labels:
        return [
            "Inspect `reports/frame_sheet.jpg` for blur, duplicates, and target coverage.",
            f"Auto-label frames: `python {script} label --target \"<target>\" --workdir \"{workdir}\" --prompt \"<specific English prompt>\" --mode balanced`.",
        ]

    steps: list[str] = []
    if stage in {"assist", "label", "mark-empty"} or best is None:
        steps.append("Inspect `reports/label_sheet.jpg`; confirm boxes cover only the target and not lookalikes.")
        if positive < 50:
            steps.append("Add more footage or increase `--sample-fps`; fewer than 50 positive frames is risky.")
        if empty == 0:
            steps.append("Add negative/background frames with `mark-empty` before training if lookalikes appear.")
        steps.append(
            f"Train after visual QA passes: `python {script} train --target \"<target>\" --workdir \"{workdir}\" --mode balanced`."
        )
        return steps

    if best is not None and not prediction_dir.exists():
        return [
            f"Run prediction QA: `python {script} evaluate --target \"<target>\" --workdir \"{workdir}\" --mode balanced --conf 0.25`."
        ]

    if prediction_dir.exists():
        return [
            "Inspect `reports/predictions/samples`; accept only if true target frames have stable boxes at normal confidence.",
            "If lookalikes are detected, clear those frames with `mark-empty`, then rerun `train` and `evaluate`.",
            "If target frames need very low confidence, add more positive frames or refine the prompt/labels.",
        ]
    return []


def label_counts(workdir: Path) -> tuple[int, int]:
    labels_dir = workdir / "labels"
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    positive = 0
    boxes = 0
    for label in labels:
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            positive += 1
            boxes += len(lines)
    return positive, boxes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-supervised YOLO training from videos.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target", default="target", help="Single object class name.")
        p.add_argument("--workdir", help="Pipeline work directory.")
        p.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu.")
        p.add_argument("--imgsz", type=int)
        p.add_argument("--conf", type=float)
        p.add_argument("--mode", choices=["fast", "balanced", "quality"], default="balanced")

    extract = sub.add_parser("extract", help="Extract and deduplicate frames from videos.")
    common(extract)
    extract.add_argument("--videos", help="Video file or folder. Defaults to E:\\Dev\\modeltraining\\videos.")
    extract.add_argument("--sample-fps", type=float)
    extract.add_argument("--max-frames", type=int)
    extract.add_argument("--jpeg-quality", type=int, default=92)
    extract.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    extract.add_argument("--dedupe-hamming", type=int, default=4)

    label = sub.add_parser("label", help="Auto-label extracted frames and generate a YOLO dataset.")
    common(label)
    label.add_argument("--prompt", help="Detection prompt. Defaults to --target.")
    label.add_argument("--label-backend", choices=["auto", "world", "yoloe"], default="auto")
    label.add_argument("--label-model", help="Open-vocabulary model name or path.")
    label.add_argument("--sam-model", help="SAM model path/name, e.g. mobile_sam.pt, sam2_t.pt, or none.")
    label.add_argument("--box-iou", type=float, default=0.40, help="Suppress duplicate boxes above this IoU after refinement.")
    label.add_argument("--val-ratio", type=float, default=0.15)
    label.add_argument("--seed", type=int, default=7)

    train = sub.add_parser("train", help="Train YOLO on the generated dataset.")
    common(train)
    train.add_argument("--train-model", default="yolo11n.pt")
    train.add_argument("--epochs", type=int)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--batch", type=parse_batch, default=-1)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--min-positive-frames", type=int)
    train.add_argument("--allow-empty-training", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Run prediction samples with the trained model.")
    common(evaluate)
    evaluate.add_argument("--weights", help="Path to best.pt. Defaults to latest workdir best.pt.")
    evaluate.add_argument("--predict-samples", type=int)

    empty = sub.add_parser("mark-empty", help="Clear selected labels so false-positive frames become negative samples.")
    common(empty)
    empty.add_argument("--images", action="append", help="Image names or comma-separated image names to clear.")
    empty.add_argument("--stems", action="append", help="Image stems or comma-separated stems to clear.")
    empty.add_argument("--first-n", type=int, help="Clear the first N extracted frames.")
    empty.add_argument("--frame-range", action="append", help="Clear frames by source frame index, e.g. 0:48 or 0-48.")
    empty.add_argument("--val-ratio", type=float, default=0.15)
    empty.add_argument("--seed", type=int, default=7)
    empty.add_argument("--dry-run", action="store_true")

    prune = sub.add_parser("prune-boxes", help="Remove selected label boxes by normalized geometry filters.")
    common(prune)
    prune.add_argument("--images", action="append", help="Image names or comma-separated image names to inspect.")
    prune.add_argument("--stems", action="append", help="Image stems or comma-separated stems to inspect.")
    prune.add_argument("--first-n", type=int, help="Inspect the first N extracted frames.")
    prune.add_argument("--frame-range", action="append", help="Inspect frames by source frame index, e.g. 0:48 or 0-48.")
    prune.add_argument("--min-xc", type=float)
    prune.add_argument("--max-xc", type=float)
    prune.add_argument("--min-yc", type=float)
    prune.add_argument("--max-yc", type=float)
    prune.add_argument("--min-width", type=float)
    prune.add_argument("--max-width", type=float)
    prune.add_argument("--min-height", type=float)
    prune.add_argument("--max-height", type=float)
    prune.add_argument("--min-area", type=float)
    prune.add_argument("--max-area", type=float)
    prune.add_argument("--val-ratio", type=float, default=0.15)
    prune.add_argument("--seed", type=int, default=7)
    prune.add_argument("--dry-run", action="store_true")

    assist = sub.add_parser("assist", help="Prepare a balanced run and pause for AI visual QA before training.")
    common(assist)
    assist.add_argument("--videos", help="Video file or folder. Defaults to E:\\Dev\\modeltraining\\videos.")
    assist.add_argument("--sample-fps", type=float)
    assist.add_argument("--max-frames", type=int)
    assist.add_argument("--jpeg-quality", type=int, default=92)
    assist.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    assist.add_argument("--dedupe-hamming", type=int, default=4)
    assist.add_argument("--prompt-samples", type=int, default=8, help="Number of clear representative frames for prompt discovery.")
    assist.add_argument("--prompt", help="Detection prompt. If omitted, assist pauses after extraction for AI prompt discovery.")
    assist.add_argument("--label-backend", choices=["auto", "world", "yoloe"], default="auto")
    assist.add_argument("--label-model", help="Open-vocabulary model name or path.")
    assist.add_argument("--sam-model", help="SAM model path/name, e.g. mobile_sam.pt, sam2_t.pt, or none.")
    assist.add_argument("--box-iou", type=float, default=0.40, help="Suppress duplicate boxes above this IoU after refinement.")
    assist.add_argument("--val-ratio", type=float, default=0.15)
    assist.add_argument("--seed", type=int, default=7)
    assist.add_argument("--train-model", default="yolo11n.pt")
    assist.add_argument("--epochs", type=int)
    assist.add_argument("--patience", type=int, default=15)
    assist.add_argument("--batch", type=parse_batch, default=-1)
    assist.add_argument("--workers", type=int, default=4)
    assist.add_argument("--min-positive-frames", type=int)
    assist.add_argument("--allow-empty-training", action="store_true")
    assist.add_argument("--predict-samples", type=int)
    assist.add_argument("--skip-extract", action="store_true")
    assist.add_argument("--skip-label", action="store_true")
    assist.add_argument("--auto-train", action="store_true", help="Continue through train and evaluate without pausing for visual QA.")

    run = sub.add_parser("run", help="Run extract, label, train, and evaluate.")
    common(run)
    run.add_argument("--videos", help="Video file or folder. Defaults to E:\\Dev\\modeltraining\\videos.")
    run.add_argument("--sample-fps", type=float)
    run.add_argument("--max-frames", type=int)
    run.add_argument("--jpeg-quality", type=int, default=92)
    run.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--dedupe-hamming", type=int, default=4)
    run.add_argument("--prompt", help="Detection prompt. Defaults to --target.")
    run.add_argument("--label-backend", choices=["auto", "world", "yoloe"], default="auto")
    run.add_argument("--label-model", help="Open-vocabulary model name or path.")
    run.add_argument("--sam-model", help="SAM model path/name, e.g. mobile_sam.pt, sam2_t.pt, or none.")
    run.add_argument("--box-iou", type=float, default=0.40, help="Suppress duplicate boxes above this IoU after refinement.")
    run.add_argument("--val-ratio", type=float, default=0.15)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--train-model", default="yolo11n.pt")
    run.add_argument("--epochs", type=int)
    run.add_argument("--patience", type=int, default=15)
    run.add_argument("--batch", type=parse_batch, default=-1)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--min-positive-frames", type=int)
    run.add_argument("--allow-empty-training", action="store_true")
    run.add_argument("--predict-samples", type=int)

    return parser


def parse_batch(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = apply_mode_defaults(parser.parse_args(argv))
    if args.command == "extract":
        workdir = extract_frames(args)
    elif args.command == "label":
        workdir = label_frames(args)
    elif args.command == "train":
        workdir = train_model(args)
    elif args.command == "evaluate":
        workdir = evaluate_model(args)
    elif args.command == "mark-empty":
        workdir = mark_empty(args)
    elif args.command == "prune-boxes":
        workdir = prune_boxes(args)
    elif args.command == "assist":
        workdir = assist_workflow(args)
    elif args.command == "run":
        workdir = extract_frames(args)
        args.workdir = str(workdir)
        label_frames(args)
        train_model(args)
        evaluate_model(args)
    else:
        parser.error(f"unknown command: {args.command}")
        return 2
    log(f"done: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
