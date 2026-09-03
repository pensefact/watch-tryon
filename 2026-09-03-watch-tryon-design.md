# Watch Virtual Try-On — Design Spec

## Overview

A tool that lets users visualize how a specific watch (by reference number) would look on their wrist at accurate physical scale. Given a watch reference, a wrist photo/video, and the user's wrist circumference, it produces a photorealistic composite showing the watch at its real-world size.

**Core value prop:** Accurate size visualization — not just "a watch on a wrist" but "this 42mm watch on your 17.5cm wrist, to scale."

**Scope:** Technical proof-of-concept. Stills first (Phase 1), video as a stretch goal (Phase 2). Production path identified (Unreal Engine + Pixel Streaming) but not in scope for POC.

## Architecture

Two independent pipelines feed into a composition engine:

```
User Input                    Watch Input
(wrist photo + circumference) (reference number)
        │                              │
        ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│ Wrist Pipeline  │          │ Watch Pipeline   │
│                 │          │                  │
│ • MediaPipe     │          │ • Scrape specs   │
│   landmark det. │          │   (diameter, L2L,│
│ • SAM 2 wrist   │          │    thickness)    │
│   segmentation  │          │ • Scrape photos  │
│ • Scale calib.  │          │ • Photo → 3D mesh│
│   from circ.    │          │   (Trellis 2)    │
│ • Lighting est. │          │ • Scale to real  │
│                 │          │   dimensions     │
└────────┬────────┘          └────────┬─────────┘
         │                            │
         ▼                            ▼
      ┌──────────────────────────────────┐
      │         Composition Engine       │
      │                                  │
      │ • Blender headless rendering     │
      │ • Camera matching                │
      │ • Shadow + occlusion compositing │
      │ • Optional diffusion refinement  │
      └──────────────┬───────────────────┘
                     │
                     ▼
              Final Output Image / Video
```

The watch pipeline results are cacheable per reference — same watch ref produces the same 3D mesh. The wrist pipeline runs per user photo.

## Watch Pipeline

### Spec Sourcing

Input: watch reference string (e.g. "Rolex 126710BLNR", "Omega 310.30.42.50.01.001")

Required dimensions:
- Case diameter (mm)
- Lug-to-lug distance (mm)
- Case thickness (mm)
- Band width (mm)
- Case shape (round / square / tonneau)

For the POC, specs are entered manually. Automated scraping (Chrono24, Watchbase, brand sites) is a later enhancement.

### Reference Photos

Minimum: one clean front-facing dial shot. Ideal: front, side profile, and angled view.

For POC: manually sourced via image search.

### Photo-to-3D Reconstruction

**Primary:** PiAPI Trellis 2 API — $0.10/generation, pay-per-use, no subscription. Best available quality for hard-surface objects.

**Comparison:** Tripo AI — 2,000 free credits on signup (~40-50 generations). Most mature developer API with good documentation.

**Fallback (if API quality insufficient):** Self-host Trellis 2 on RunPod A100 (~$1.50/hr). MIT licensed.

### Mesh Post-Processing

1. Scale the reconstructed mesh so its bounding box matches the scraped dimensions (diameter → X, lug-to-lug → Y, thickness → Z)
2. Orient consistently (dial face = +Z, 12 o'clock = +Y)
3. Fallback for poor-quality meshes: project the front-facing reference photo as a texture onto a simplified watch-shaped primitive (cylinder + lugs). Guarantees correct proportions even if reconstruction fails.

## Wrist Pipeline

### Detection

**MediaPipe Hands** detects 21 hand landmarks. Landmark 0 (wrist base) locates the wrist region. Landmarks 5 (index MCP) and 17 (pinky MCP) give forearm direction.

MediaPipe provides point locations only, not segmentation masks.

### Segmentation

**SAM 2** (Segment Anything Model 2) takes the MediaPipe wrist landmark as a point prompt and returns a pixel-precise wrist/forearm mask.

From the mask:
- Extract wrist contour at the watch-wearing point
- Measure contour width perpendicular to the forearm axis
- This gives actual pixel width of the wrist

### Scale Calibration

1. SAM 2 mask provides the apparent wrist width in pixels
2. User-provided circumference (e.g. 17.5cm) → approximate wrist as an ellipse
3. Wrist pose angle (from MediaPipe landmarks) determines which cross-section of the ellipse is visible
4. Solve for **pixels-per-mm ratio** at the wrist surface

Example: A 42mm watch on a 17.5cm wrist covers ~24% of the circumference. If the wrist appears 200px wide at the measured angle, the watch case should span a computable number of pixels.

### Lighting Estimation

- **POC approach:** Detect the brightest region in the image, assume a single dominant light source from that direction. Watches are reflective — even rough directional lighting sells the illusion.
- **Upgrade path:** DiffusionLight or spherical harmonics estimator for a full environment map.

### Pipeline Output

A placement specification:
- Wrist center (x, y) in image coordinates
- Rotation angle (degrees)
- Scale factor (pixels per mm)
- Estimated environment lighting
- Wrist segmentation mask (for occlusion — parts of hand/wrist that appear in front of the watch band)

## Composition Engine (Phase 1 — Stills)

### 3D Rendering

**Blender Python (bpy) headless:**

1. Load the scaled watch mesh
2. Set up a virtual camera matching the input photo's perspective:
   - Position/angle from wrist pose estimation
   - Focal length from EXIF data or assumed ~28mm (typical phone camera)
3. Apply estimated lighting
4. Render three passes:
   - **Color pass** — textured watch
   - **Shadow pass** — soft shadow cast on wrist
   - **Mask pass** — silhouette for compositing

### Compositing

1. Place watch color pass at wrist center, rotated to forearm angle, scaled per pixels-per-mm ratio
2. Use wrist segmentation mask for occlusion (thumb side wrapping over band)
3. Multiply shadow pass onto wrist skin beneath the watch
4. Feather watch edges to avoid hard cutout look

### Optional Diffusion Refinement

If the composite looks "pasted on":
- Stable Diffusion XL inpainting with the composite as input
- Mask only a thin border around watch edges
- Low denoising strength (0.2–0.3) — harmonize lighting/reflections without changing watch identity or size
- Watch geometry stays locked from the 3D render

## Video Extension (Phase 2)

### Wrist Tracking

SAM 2 in **video mode**: provide a point prompt on frame 1, SAM 2 propagates the segmentation mask across all subsequent frames automatically. Handles occlusions and maintains identity.

### Placement Smoothing

Raw per-frame measurements will jitter. Apply a **1-Euro filter or Kalman filter** on placement parameters (position, angle, scale) for smooth tracking that still follows natural wrist motion.

### Rendering Strategy

**Per-frame 3D render** in Blender. At ~100ms per frame on GPU, a 5-second clip (150 frames at 30fps) takes ~15 seconds. Handles wrist rotation correctly — no perspective distortion artifacts from warping.

### Temporal Coherence

- Lock lighting estimate from frame 1 (or average of first few frames), hold constant across clip
- Consistent shadow parameters across frames
- No per-frame diffusion refinement — destroys temporal consistency. Either apply to keyframes and interpolate, or skip for video.

### Video Pipeline

```
Frame 1:
  MediaPipe → wrist landmark
  SAM 2 video mode → masks for ALL frames
  Lighting estimation → locked for clip

Per frame:
  Extract placement from mask (smoothed)
  Blender render watch at frame's perspective
  Composite onto frame

Encode frames → output video (ffmpeg)
```

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python | All pipeline components have Python SDKs |
| 3D Reconstruction | PiAPI (Trellis 2) + Tripo for comparison | Hosted APIs, no local GPU needed for this step |
| Wrist Detection | MediaPipe Hands | Free, fast, local |
| Wrist Segmentation | SAM 2 | Point-prompt from MediaPipe, video mode for Phase 2 |
| Depth (if needed) | Depth Anything V2 | Helps with angled wrist shots |
| 3D Rendering | Blender Python (bpy) headless | Best material support for metals/glass, scriptable |
| Compositing | OpenCV + Blender render passes | Shadow blending, edge feathering, mask occlusion |
| Diffusion refinement | SDXL inpainting (optional) | Edge cleanup only |
| Video encoding | ffmpeg | Standard |
| POC Interface | Gradio | Shareable web UI with minimal effort |

## Runtime Environment

Single machine with GPU:
- Local machine with GPU, or
- Cloud GPU instance (RunPod, Lambda, Vast.ai) at ~$0.50–1.50/hr
- Google Colab Pro as a quick option

## Estimated Costs

| Item | Cost |
|------|------|
| 3D reconstruction API calls (~50 iterations) | ~$5–25 |
| Cloud GPU time (~10 hours dev/testing) | ~$15 |
| Everything else | Free / open source |
| **Total** | **~$20–40** |

## Estimated Latency

### Phase 1 — Stills

| Step | Time |
|------|------|
| Watch spec input (manual) | — |
| Photo-to-3D (PiAPI) | 10–30s |
| MediaPipe + SAM 2 | 2–5s |
| Blender render | 3–10s |
| Compositing | <1s |
| Diffusion refinement (optional) | 5–15s |
| **Total** | **~20–60s** |

3D reconstruction is cacheable per watch ref — repeat tries with different wrist photos skip this step.

### Phase 2 — Video (5-second clip, 30fps)

| Step | Time |
|------|------|
| Watch pipeline (cached) | 0s |
| SAM 2 video propagation | 5–10s |
| Placement extraction + smoothing | <1s |
| 150× Blender renders (GPU) | 15–30s |
| 150× compositing | 2–5s |
| Video encoding | 1–2s |
| **Total** | **~25–50s** |

## Production Upgrade Path (Out of Scope for POC)

If the POC validates, the rendering layer would migrate to **Unreal Engine + Pixel Streaming**:
- Real-time rendering at 60fps with Lumen GI and ray-traced reflections
- Superior watch material rendering (brushed steel, sapphire crystal, ceramic)
- Composure framework for real-time CG-on-live-footage compositing
- Pixel Streaming: Unreal renders on cloud GPU, streams to browser — no install, works on any device
- Opens the door to interactive webcam-driven live try-on

The rest of the pipeline (watch data sourcing, wrist detection, segmentation, scale calibration) carries over unchanged.
