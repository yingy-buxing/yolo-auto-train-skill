# YOLO Auto Train Skill

A Codex skill for training a single-class YOLO object detector from user-provided videos.

The skill is built around a practical AI-assisted loop:

1. Sample a few clear frames for prompt discovery.
2. Use a vision-language detector to auto-label the target object.
3. Optionally refine boxes with a SAM-compatible model.
4. Generate a YOLO-format dataset.
5. Inspect contact sheets before training.
6. Train with Ultralytics YOLO.
7. Run prediction samples and decide whether labels need another cleanup pass.

It is intended for small, focused object detectors where the user can provide one or more videos of the object they care about.

## Repository Layout

```text
.
+-- SKILL.md
+-- agents/
|   +-- openai.yaml
+-- scripts/
|   +-- yolo_auto_train.py
+-- requirements.txt
+-- README.md
```

## Install As A Codex Skill

Clone this repository, then copy or link the folder into your Codex skills directory:

```powershell
git clone https://github.com/yingy-buxing/yolo-auto-train-skill.git
Copy-Item -Recurse -Force .\yolo-auto-train-skill "$env:USERPROFILE\.codex\skills\yolo-auto-train"
```

Install Python dependencies:

```powershell
pip install -r "$env:USERPROFILE\.codex\skills\yolo-auto-train\requirements.txt"
```

Optional SAM refinement uses Ultralytics-compatible SAM weights such as `mobile_sam.pt` or `sam2_t.pt`.

## Quick Start

Put one or more target videos in a folder, then run prompt discovery:

```powershell
python "$env:USERPROFILE\.codex\skills\yolo-auto-train\scripts\yolo_auto_train.py" assist `
  --target "dog" `
  --videos "D:\videos\dog" `
  --mode balanced
```

Without `--prompt`, the command only extracts a few clear representative frames and writes:

```text
reports/prompt_sheet.jpg
```

Inspect that image, write an English visual prompt for the target, then continue:

```powershell
python "$env:USERPROFILE\.codex\skills\yolo-auto-train\scripts\yolo_auto_train.py" assist `
  --target "dog" `
  --videos "D:\videos\dog" `
  --workdir "D:\runs\dog-balanced" `
  --prompt "light colored dog" `
  --mode balanced
```

When labels look good, train and evaluate:

```powershell
python "$env:USERPROFILE\.codex\skills\yolo-auto-train\scripts\yolo_auto_train.py" train `
  --target "dog" `
  --workdir "D:\runs\dog-balanced"

python "$env:USERPROFILE\.codex\skills\yolo-auto-train\scripts\yolo_auto_train.py" evaluate `
  --target "dog" `
  --workdir "D:\runs\dog-balanced"
```

## Modes

- `fast`: quick feasibility test, fewer frames, no SAM by default.
- `balanced`: recommended default, more frames, optional MobileSAM refinement.
- `quality`: slower and larger run, more frames and higher image size.

## Important Outputs

Inside each run directory:

- `prompt_frames/`: a few clear frames for prompt writing.
- `frames/`: extracted training frames.
- `dataset/`: YOLO-format training dataset.
- `reports/prompt_sheet.jpg`: prompt discovery contact sheet.
- `reports/frame_sheet.jpg`: extracted-frame QA sheet.
- `reports/label_sheet.jpg`: auto-label QA sheet.
- `reports/predictions/`: post-training prediction samples.
- `train/`: Ultralytics training output.

## Notes

- This repository intentionally does not include videos, generated datasets, training runs, or model weights.
- For best results, keep prompts target-specific and avoid broad context words. For example, `light colored dog` is usually better than `dog running through river near person`.
- Treat the generated labels as AI-assisted drafts. Inspect contact sheets before training.

## License

MIT
