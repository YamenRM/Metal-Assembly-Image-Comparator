# AI Metal Assembly Image Comparator

A lightweight computer-vision tool that compares two photos of a metal assembly (a reference and a target) and automatically highlights any structural differences — mis-welds, missing brackets, misaligned components, etc. — on a side-by-side view.

---

## How It Works

The tool runs three steps in sequence:

1. **Alignment** (`utils/alignment.py`) — Detects SIFT keypoints in both images and computes a homography (via RANSAC) to warp the target image so it matches the reference image's perspective. This corrects for the camera being held at a slightly different angle/position between the two shots. It also returns a *validity mask* marking which pixels in the warped image are real photo content vs. empty space introduced by the warp.
2. **Comparison** (`utils/comparator.py`) — Converts both images to grayscale, applies CLAHE (adaptive contrast normalization) to reduce the impact of glare and reflections common on stainless/metallic surfaces, then applies a light Gaussian blur to absorb the small (sub-pixel) registration noise that any camera-angle correction leaves behind — without this, that noise alone can get misread as a defect along every edge and bolt outline in the image. It then runs a Structural Similarity (SSIM) comparison to find regions that differ. The validity mask from step 1 ensures warp edges aren't mistaken for real defects. Small artifacts below a minimum area are filtered out.
3. **Visualization** (`utils/visualizer.py`) — Draws bounding boxes and a semi-transparent highlight over each detected difference on the target image, then places it next to the original reference image for side-by-side inspection.

---

## Requirements

- Python 3.9 or newer
- Dependencies:
  ```
  opencv-python
  scikit-image
  numpy
  streamlit
  Pillow
  ```

Install everything with:

```bash
pip install Requirements.txt
```
---

## Project Structure

```
project/
├── app.py                  # Streamlit application (main entry point)
├── utils/
│   ├── __init__.py
│   ├── alignment.py         # Step 1: image alignment
│   ├── comparator.py        # Step 2: difference detection
│   └── visualizer.py        # Step 3: side-by-side highlighted output
└── README.md
```

**Important:** `alignment.py`, `comparator.py`, `visualizer.py`, and `__init__.py` must live inside a folder named `utils/`, sitting next to `app.py`. This is required for the imports in `app.py` to resolve.

---

## Running the App

From the project's root folder:

```bash
streamlit run app.py
```

This opens the tool in your browser (usually `http://localhost:8501`). No installation of a separate desktop executable is needed — it runs locally on your machine and does not send images anywhere.

---

## Using the Tool

1. **Upload Image A (Reference)** — the "known good" photo of the assembly.
2. **Upload Image B (Target)** — the photo you want to check for defects.
3. Adjust the two tuning sliders if needed (defaults work well for most cases — see below).
4. Click **Compare Images**.
5. The result appears below: the reference photo on the left, and the target photo on the right with any detected differences boxed and highlighted in orange/red.

Supported formats: JPG, JPEG, PNG, GIF, BMP.

### Tuning Parameters

| Parameter | What it does | When to adjust |
|---|---|---|
| **Detection Sensitivity (SSIM Threshold)** | Controls how different two regions must be before they're flagged. Lower = more sensitive (catches subtler defects, but may flag more false positives from lighting/glare). Higher = stricter (fewer false alarms, but may miss subtle defects). | Increase if you're seeing too many flags on surface scratches, dust, or reflections. Decrease if a known defect isn't being caught. |
| **Minimum Feature Size (pixels)** | Filters out tiny flagged regions below this area. Prevents noise/glare specks from cluttering the result. | Increase if small reflections or JPEG-compression artifacts are showing up as false positives. Decrease if a small but real defect (e.g. a hairline gap) is being filtered out. |

---

## Notes on Accuracy & Tuning

- The default settings (`ssim_thresh=0.45`, `min_area=50`) are a reasonable starting point but were tuned on limited sample data. **Before relying on this for production defect sign-off, run it against a representative batch of your own real shop-floor photos and adjust the two sliders until it reliably catches known defects without excessive false flags.**
- The pipeline includes a small blur step specifically to prevent normal camera-angle correction from being mistaken for a defect (see "How It Works" above). If you still see widespread flagging across an otherwise-identical image pair, try nudging **Detection Sensitivity** down slightly before assuming there's a real discrepancy.
- Very strong, sharp reflections (direct light bouncing straight into the lens) can still occasionally be flagged as differences — CLAHE reduces this but doesn't eliminate it entirely. If this becomes a recurring issue on your specific parts/lighting setup, let us know — the sensitivity can be tuned further or a reflection-specific filter can be added.
- Processing time is roughly 1–3 seconds per image pair on a standard laptop CPU (no GPU required).

---

## Support

For threshold tuning, adding new file format support, or extending the tool (e.g., batch processing multiple pairs, exporting a report), the code is organized so each step (`alignment.py`, `comparator.py`, `visualizer.py`) can be modified independently — see inline comments in each file for guidance.
