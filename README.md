# watch-tryon

A proof-of-concept for virtual watch try-on using computer vision. Upload a photo of your wrist and a watch, and the system composites the watch onto the wrist with correct scale, perspective, and lighting.

## Pipeline

The still-image pipeline runs these steps in sequence:

1. **Wrist detection** — locates the wrist in the input photo.
2. **Segmentation** — isolates the wrist region (used for occlusion and blending).
3. **Calibration** — computes pixels-per-mm from the wrist circumference to size the watch correctly.
4. **Lighting estimation** — estimates the light direction from the wrist image.
5. **Watch reconstruction** — generates a 3D mesh of the watch from its product photo.
6. **Mesh processing** — loads and scales the mesh to match the wrist dimensions.
7. **Rendering** — renders the watch mesh with matched lighting.
8. **Compositing** — blends the rendered watch onto the wrist image.

A video pipeline adds frame-by-frame wrist tracking on top of the same steps.

## UI

A Gradio interface lets you upload a wrist photo and watch photo, enter your wrist circumference and watch dimensions (case diameter, lug-to-lug, thickness, band width, case shape), and see the result.

## Setup

```bash
pip install -r requirements.txt
python run_demo.py
```

Requires an API key in `.env` for the 3D reconstruction step (see `.env.example`).

## Project structure

```
src/
  pipeline.py              # Main still-image pipeline
  app.py                   # Gradio UI
  models.py                # WatchSpec, PlacementSpec dataclasses
  wrist_pipeline/          # Detection, segmentation, calibration, lighting
  watch_pipeline/          # 3D reconstruction, mesh processing
  composition/             # Rendering and compositing
  video/                   # Video pipeline with tracking
notebooks/
  colab_demo.ipynb         # Google Colab notebook
```

## Requirements

- Python 3.11+
- OpenCV, NumPy, Pillow, Gradio
