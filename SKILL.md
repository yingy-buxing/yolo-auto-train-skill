---
name: yolo-auto-train
description: Automate single-class YOLO object-detection training from user-provided videos. Use when the user wants to train a YOLO model for a named target object from video footage, especially with AI-assisted frame selection, open-vocabulary auto-labeling, SAM/SAM2-style mask refinement, multimodal QA, dataset generation, and local Ultralytics training.
---

# YOLO Auto Train

## Purpose

Use this skill to turn videos of one target object into a trained YOLO detection model with an AI-supervised pipeline:

1. Extract and deduplicate frames.
2. Auto-label the named target with open-vocabulary vision models.
3. Optionally refine boxes through SAM-compatible segmentation.
4. Generate YOLO-format data.
5. Review visual contact sheets before training.
6. Train, evaluate, and report results.

The script does deterministic batch work. Codex provides the multimodal judgment: inspect previews, adjust prompts/thresholds, decide whether to proceed, and explain quality risks.

## Default Workflow

Start every real user run by collecting:

- Target object name.
- Video file or folder path. If missing, ask the user to place videos in a local `videos/` folder or provide an absolute path.
- Optional target description or reference phrase in English when the target is a common visual class.

Use the bundled CLI from this skill. If no strong prompt is already known, start with lightweight prompt discovery:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" assist --target "target object" --videos ".\videos" --mode balanced
```

When `assist` has no `--prompt`, it extracts only a few clear representative frames into `prompt_frames/`, writes `reports/prompt_sheet.jpg`, and pauses before labeling. Open `reports/prompt_sheet.jpg`, inspect the target and distractors, then write a specific English prompt for the target in this particular video. Continue with a full extraction and label run:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" assist --target "target object" --videos ".\videos" --workdir ".\runs\<run>" --prompt "<specific visual prompt>" --mode balanced
```

Default assumptions:

- Single-class object detection.
- Local Ultralytics training.
- Work directory: `YOLO_AUTO_TRAIN_ROOT\runs\<target>-<timestamp>` when `YOLO_AUTO_TRAIN_ROOT` is set, otherwise a local `yolo-auto-train-runs/` directory.
- Auto-label backend: Ultralytics open-vocabulary model.
- `balanced` and `quality` try SAM-compatible refinement by default; if unavailable or unreasonable, the script falls back to detector boxes.

## Decision Loop

Do not treat the pipeline as a blind one-shot command. After each visual stage:

- Before first auto-labeling, use `reports/prompt_sheet.jpg` to write the detection prompt. Describe the visible target, not just the class name.
- Open `reports/frame_sheet.jpg` after full extraction. Reject frames that are mostly blurry, duplicated, or target-free unless they are intended negatives.
- Open `reports/label_sheet.jpg` after auto-labeling. Check whether boxes cover the requested target and not lookalikes.
- If labels are poor, rerun `label` with a better `--prompt`, adjusted `--conf`, or `--sam-model`.
- If lookalike frames are mislabeled, use `mark-empty` to turn them into negative/background examples, then inspect `label_sheet.jpg` again.
- Start training only after label previews are plausible.
- After training, inspect prediction samples in `reports/predictions/` and summarize likely failure modes.

When the user insists on full automation, still generate previews and continue, but call out quality warnings in the final report.

## Commands

Run the full pipeline:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" run --target "screwdriver" --videos ".\videos" --mode balanced
```

Prepare a balanced run with prompt discovery, then pause before training for AI visual QA:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" assist --target "screwdriver" --videos ".\videos" --mode balanced
python "$skill\scripts\yolo_auto_train.py" assist --target "screwdriver" --videos ".\videos" --workdir ".\runs\screwdriver-20260519-120000" --prompt "red handled screwdriver" --mode balanced
```

Continue through train/evaluate without pausing only when the user explicitly wants full automation:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" assist --target "screwdriver" --videos ".\videos" --prompt "red handled screwdriver" --mode balanced --auto-train
```

Extract only:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" extract --target "screwdriver" --videos ".\videos" --sample-fps 1 --max-frames 1200
```

Auto-label an existing frame folder:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" label --target "screwdriver" --workdir ".\runs\screwdriver-20260519-120000"
```

Clear false-positive frames as negative/background samples:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" mark-empty --target "screwdriver" --workdir ".\runs\screwdriver-20260519-120000" --images v000_f00000000.jpg,v000_f00000002.jpg
```

Clear a source frame-index range as negatives:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" mark-empty --target "screwdriver" --workdir ".\runs\screwdriver-20260519-120000" --frame-range 0:48
```

Remove obvious bad boxes while keeping other boxes in the same frame:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" prune-boxes --target "dog" --workdir ".\runs\dog-run" --images frame001.jpg,frame002.jpg --max-xc 0.20 --min-height 0.90
```

Train an existing generated dataset:

```powershell
$skill = "$env:USERPROFILE\.codex\skills\yolo-auto-train"
python "$skill\scripts\yolo_auto_train.py" train --workdir ".\runs\screwdriver-20260519-120000"
```

## Modes

Mode changes real pipeline defaults unless the user explicitly overrides a flag:

- `fast`: quick feasibility. `sample_fps=0.5`, `max_frames=300`, `conf=0.20`, `imgsz=640`, no SAM, `epochs=25`, minimum 20 positive frames.
- `balanced`: default practical run. `sample_fps=1.0`, `max_frames=1200`, `conf=0.10`, `imgsz=640`, default `--sam-model mobile_sam.pt`, `epochs=50`, minimum 50 positive frames.
- `quality`: higher-effort run. `sample_fps=2.0`, `max_frames=3000`, `conf=0.05`, `imgsz=960`, default `--sam-model mobile_sam.pt`, `epochs=100`, minimum 120 positive frames.

Use `fast` first for a new target/video, then rerun `balanced` or `quality` after the frame and label previews look plausible. `quality` may download SAM weights and is slower.

SAM refinement is guarded: refined boxes replace detector boxes only when area and center-shift checks look reasonable. Otherwise the detector box is kept. Check `sam_refined_boxes`, `sam_fallback_boxes`, and `sam_unavailable_boxes` in `reports/report.md`.

After detection/refinement, the script suppresses duplicate same-class boxes above `--box-iou 0.40` by default. This reduces double-label noise on a single object while preserving separated objects.

## Prompt Rules

Open-vocabulary prompts decide what gets labeled. Keep them specific:

- Good: `spotted cheetah`, `red handled screwdriver`, `white ceramic mug`.
- Risky: `animal`, `tool`, `object`, `thing`, or broad prompt phrases that include lookalikes.
- For Chinese target names, keep `--target` in Chinese if desired, but use an English `--prompt` when the detector works better that way.
- If a prompt captures lookalikes, narrow the prompt and add false-positive frames as empty labels with `mark-empty`.

Prompt discovery rule:

- If the user only gives a target name, run prompt discovery first and inspect `prompt_sheet.jpg`.
- Prompt discovery should be lightweight: use `reports/prompt_sheet.jpg`, usually 6-10 clear frames, only to understand the target appearance. Do not extract hundreds of frames just to write the prompt.
- Write the prompt from visible evidence: color, shape, material, pose, viewpoint, size, and nearby distractors.
- Prefer target-appearance prompts such as `light colored dog` over broad context prompts such as `dog running through river near person`.
- Do not include distractors in the prompt unless excluding them by narrowing the target; avoid broad context like `person with dog` because open-vocabulary detectors may label both.

## Multimodal QA Rules

Use visual inspection whenever contact sheets exist. Look for:

- The target object is visible and varied across angles/backgrounds.
- Boxes are tight enough for detector training.
- The model is not consistently selecting a nearby lookalike.
- Empty-label frames are reasonable negatives, not systematic misses.
- The dataset has enough positives. Fewer than 50 labeled frames is a high-risk dataset.
- The dataset has some negative/background frames when lookalikes appear in the video.

If quality is bad, prefer one rerun before training:

- Improve prompt: add color, material, shape, brand, part name, or exclusion words.
- Raise `--conf` when false positives dominate.
- Lower `--conf` when the target is frequently missed.
- Enable or change `--sam-model` when boxes are loose.
- Disable SAM with `--sam-model none` if refinement produces worse labels.
- Extract more or different footage when scenes are too repetitive.
- Use `mark-empty` for frames where the detector confidently labeled a lookalike.
- Use `prune-boxes` when one bad box appears in a frame that also contains a useful target box.

## Training Acceptance

After training, do not rely only on mAP. Inspect prediction images:

- Accept when true target frames have one stable, tight box at normal confidence such as `0.25`.
- Reject or revise when target frames need very low confidence such as `0.01`.
- Reject or revise when lookalikes or backgrounds get boxes at normal confidence.
- If `best.pt` has good metrics but poor sample predictions, test `last.pt`; then report which weight behaves better.

## Outputs

Key outputs inside the work directory:

- `frames/`: extracted source frames.
- `dataset/images/{train,val}/`: YOLO images.
- `dataset/labels/{train,val}/`: YOLO labels.
- `dataset/data.yaml`: Ultralytics dataset config.
- `reports/frame_sheet.jpg`: extraction preview.
- `reports/label_sheet.jpg`: auto-label preview.
- `reports/report.md`: pipeline summary, warnings, and recommended next commands.
- `train/`: Ultralytics training output.

## Dependencies

Expected Python packages:

- `ultralytics`
- `torch`
- `opencv-python`
- `Pillow`
- `PyYAML`
- `numpy`

Optional packages/models:

- Ultralytics `SAM` / `sam2_*.pt` / `mobile_sam.pt` for box-to-mask refinement.
- External Grounded-SAM2 or GroundingDINO installs for future stronger backends.

If optional backends are missing, use detector-only labels and clearly report that SAM refinement was skipped.
