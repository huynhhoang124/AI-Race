---
name: viettel-ai-race-bts
description: Guidelines, strict formatting checklists, and optimization strategies for the Viettel AI Race (Digital Twin BTS) competition to guarantee 100% submission validity and maximize metrics.
---

# Viettel AI Race - Digital Twin BTS Skill

This skill file enforces the rules, constraints, data specifications, and metrics of the Viettel AI Race (Digital Twin BTS) competition. Any AI agent working on this repository must read and strictly adhere to these guidelines to ensure the submission is valid and optimized for a top leaderboard score.

---

## 🎯 1. Mission & Objectives
* **Goal**: Reconstruct the 3D structure of BTS telecommunication towers (Digital Twin) from drone/hand-held camera images, and synthesize RGB images at unseen/novel camera viewpoints.
* **Core Metric**: 
  $$\text{Score} = 0.4 \times (1 - \text{LPIPS}) + 0.3 \times \text{SSIM} + 0.3 \times \text{PSNR}_{\text{norm}}$$
  * $\text{PSNR}_{\text{norm}} = \operatorname{clamp}(\text{PSNR\_val} / \text{PSNR\_max}, 0.0, 1.0)$
  * Prioritize **LPIPS** (40% weight) by avoiding blurry borders and leveraging LPIPS loss during training.

---

## 📂 2. Data Structure & Input Formats

### Dataset Directory Layout
```text
├── train/
│   ├── images/          : Training images (~80% of data)
│   └── sparse/0/        : COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
└── test/
    └── test_poses.csv   : Target camera poses for test images (~20% of data)
```

### Camera Poses Format (`test_poses.csv`)
Each row represents a target camera configuration for rendering:
* `image_name`: Filename of the target image to render (e.g. `0001.png`).
* `qw, qx, qy, qz`: Quaternion rotation matching COLMAP orientation format.
* `tx, ty, tz`: Camera translation vector.
* `fx, fy`: Camera focal lengths.
* `cx, cy`: Principal points (optical center coordinates).
* `width, height`: Target image dimensions.

---

## 🗂️ 3. Submission Integrity Checklist
All rendered images must be packaged into a single ZIP archive named `submission.zip` matching this layout:
```text
submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
```
> [!IMPORTANT]
> **Zero Tolerance Compliance Policy:**
> * Exact folder names matching target scene IDs.
> * Exact file names (`.png` format).
> * Exact image dimensions (`width` x `height`) matching `test_poses.csv`.
> * Exact file count. **Any missing, extra, or incorrectly sized file will result in an automatic 0 score for the entire submission.**

---

## 💎 4. Key Optimization Guidelines
To achieve a top leaderboard position, implement the following techniques:
1. **Asymmetric Projection Matrix**:
   Use camera-specific `cx` and `cy` (rather than symmetric $\frac{w}{2}, \frac{h}{2}$) in the projection matrix calculation (`utils/graphics_utils.py` and `scene/cameras.py`) to eliminate sub-pixel translation misalignment.
2. **LPIPS Loss Integration**:
   Combine $L_1$ and SSIM loss with a pre-trained VGG LPIPS loss function in the fine-tuning training loop (iterations > 20,000) to keep edges sharp.
3. **Histogram Matching**:
   Run histogram color matching post-processing on test renders targeting the nearest training view's color distribution to eliminate global exposure drifts.
4. **Depth Regularization**:
   Leverage aligned monocular depth maps (generated via Depth Anything v2) to constrain thin structures and eliminate floaters.

---

## 🛑 5. Anti-Cheating & Reproducibility Rules
* **No External Data**: Do not scrape or use external images/videos of the same target scenes.
* **No Manual Edits**: Do not Photoshop, manually crop, or modify individual image outputs. Everything must be generated fully automatically by the pipeline scripts.
* **Full Reproducibility**: Maintain clean configs, environment files (`environment.yml`), model checkpoints, and logs to allow the organizers to reproduce all results.
