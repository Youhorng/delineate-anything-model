# DelAny Fine-Tuning: Data Preparation & Hardware Guide

**Custom dataset transfer learning for agricultural field boundary detection**

This guide covers how to prepare a custom dataset and fine-tune **Delineate Anything (DelAny)** starting from `DelineateAnything.pt`. It is written for workflows like the Cambodia / Amret pilot: ~32 tiles (~25 km² each), Google Earth Pro (~1 m) imagery, **GEP export → Photoshop crop → KML-based GeoTIFF**, DelAny zero-shot → manual polygon cleanup → fine-tune → DelAnyFlow inference.

---

## Table of Contents

1. [Glossary & Core Concepts](#1-glossary--core-concepts)
2. [What DelAny Does & What Fine-Tuning Changes](#2-what-delany-does--what-fine-tuning-changes)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Project Overview & End-to-End Pipeline](#4-project-overview--end-to-end-pipeline)
5. [Phase 1: Export from GEP & Crop in Photoshop](#5-phase-1-export-from-gep--crop-in-photoshop)
6. [Phase 2: Create GeoTIFF from KML + Cropped Image](#6-phase-2-create-geotiff-from-kml--cropped-image)
7. [Phase 3: Alignment QA](#7-phase-3-alignment-qa)
8. [Phase 4: Chip Imagery & Build YOLO Labels](#8-phase-4-chip-imagery--build-yolo-labels)
9. [Phase 5: Dataset Layout & Train/Val Split](#9-phase-5-dataset-layout--trainval-split)
10. [Phase 6: Transfer Learning (Fine-Tuning)](#10-phase-6-transfer-learning-fine-tuning)
11. [Phase 7: Inference with Fine-Tuned Weights](#11-phase-7-inference-with-fine-tuned-weights)
12. [Troubleshooting](#12-troubleshooting)
13. [Checklists](#13-checklists)

---

## 1. Glossary & Core Concepts

### DelAny Core

**DelAny** is a **YOLOv11 instance-segmentation** network trained to answer:

> *How many separate agricultural fields are in this 512×512 RGB patch, and what is the binary mask of each?*

It is **not**:
- Semantic segmentation (field vs non-field pixels with no per-field separation)
- Boundary/edge detection only (lines without filled regions)
- Plain object detection (bounding boxes only, no shape)

It **is**:
- A neural network that **detects each field as its own separate object** and **draws a mask around it**

**Pipeline inside the model (one 512×512 patch):**

```
Input: 512×512×3 RGB patch
        ↓
Backbone + neck: turn pixels into features
        (patterns learned from COCO + FBIS-22M)
        ↓
Detect: find each field as a separate object
        - Bounding box: rectangle around the field
        - Confidence: how sure the model is
        - Class: what it is (only "field" for DelAny)
        Detection does NOT know the exact field shape inside the box.
        ↓
Segment: for each detection, predict one mask
        - Which exact pixels belong to this field?
        - For each field, paint the exact pixel mask of its shape.
```

**Training history:**
- Initialized from **COCO**-pretrained YOLOv11-seg
- Trained on **FBIS-22M** (~637k train images, ~23M field instances, Europe, 0.25–10 m GSD)
- Published weights: [`DelineateAnything.pt`](https://huggingface.co/MykolaL/DelineateAnything)

---

### Transfer Learning

**Transfer learning** means:

1. Start from `DelineateAnything.pt` (already knows generic field patterns from FBIS-22M)
2. Fine-tune on your **custom dataset** (local imagery + cleaned field polygons)
3. Produce a new `best.pt` adapted to your region, imagery source, and field sizes

You are **not** training from scratch. You are **adapting** an existing model with a low learning rate on a smaller local dataset.

---

### Raster vs Vector

| Term | Meaning | Examples |
|------|---------|----------|
| **Raster** | Grid of pixels with values | GeoTIFF, PNG, JPEG |
| **Vector** | Shapes as points/lines/polygons | GPKG, Shapefile, KML, GeoJSON |

DelAny **reads rasters** at inference. Labels start as **vectors** and are converted to **pixel polygons** for training.

---

### Tile vs Patch / Chip

| Term | Size | Role |
|------|------|------|
| **Tile** | ~25 km² (e.g. 5 km × 5 km) | One large working area |
| **Patch / Chip** | **512×512 pixels** | One cutout fed to the neural network |

One tile produces hundreds of chips for training.

---

### GSD (Ground Sampling Distance)

**GSD** = meters per pixel on the ground.

- Google Earth Pro imagery: typically ~**0.8–1.5 m** per pixel
- Sentinel-2: ~**10 m** per pixel (often too coarse for small fields)

DelAny is resolution-agnostic, but fine-tuning on ~1 m imagery teaches it your local field scale.

---

### CRS (Coordinate Reference System)

**CRS** = the coordinate system for map data.

- Example for Cambodia: **EPSG:32648** (UTM zone 48N)
- GeoJSON/KML may use WGS84 (EPSG:4326) or a projected CRS

**Rule:** Raster and vector layers must use a **consistent CRS** (or be reprojected) before chipping.

---

### Georeferencing

**Georeferencing** links **image pixels** to **real map coordinates**.

Without georeferencing:
- A PNG is only a color grid — no Earth location
- Polygons in KML/GeoJSON exist in map space with **no defined link** to PNG pixels

With georeferencing (GeoTIFF or PNG + world file):
- You can convert: **map corner → pixel (col, row)**
- This is required to build correct YOLO `.txt` labels from map-coordinate polygons

**Important:** Georeferencing is needed for **label preparation** and **DelAnyFlow inference**, not for the Ultralytics trainer itself (which only reads PNG pixels + `.txt`).

---

### Mask

A **mask** is a binary (or soft) image for **one field** — a **filled shape**, not just a border line.

YOLO seg stores this as a **polygon** (vertex list). Ultralytics rasterizes it internally for training loss.

---

### Mosaic (Training Augmentation Only)

**Mosaic** is a **training-time** YOLO augmentation — **not** used at inference.

- Takes **random crops from 4 training images**
- Stitches them into **one fixed 512×512** image
- Adjusts labels to match
- Helps the model see more fields and scales per training step

This is **not** the same as mosaicking your 32 country tiles into one image.

---

### Intersection & Union (Evaluation)

- **Intersection:** pixels where prediction **and** ground truth both say "this field"
- **Union:** pixels where **either** prediction or ground truth (or both) say "this field"

**IoU = Intersection / Union** — higher means better boundary agreement.

---

## 2. What DelAny Does & What Fine-Tuning Changes

### Official training data (FBIS-22M)

| Property | Value |
|----------|--------|
| Images | ~673k patches |
| Instances | ~22.9M field masks |
| Train / test | ~636,784 / ~36,125 |
| GSD | 0.25 m – 10 m |
| Labels | LPIS cadastre + manual correction |
| Format | YOLO instance seg: images + `.txt` polygon labels |

FBIS authors had **georeferenced imagery + map-coordinate polygons**. They converted to **pixel YOLO labels** during dataset prep. The published HuggingFace dataset is **already chipped** — you download JPG + `.txt`, not raw GeoTIFF + LPIS.

### Your workflow (same destination, earlier in the pipeline)

| Stage | FBIS-22M | Your custom dataset |
|-------|----------|---------------------|
| Imagery | Georeferenced satellite/aerial | GEP PNG → Photoshop crop → **KML-georeferenced GeoTIFF** |
| Labels | Map polygons (LPIS) | DelAny output → **cleaned KML/GeoJSON** |
| Prep | Chip + map→pixel → `.txt` | **You perform this step** |
| Train | Ultralytics | **Same** |
| Output | `DelineateAnything.pt` | `best.pt` |

### Why GeoTIFF is needed to build `.txt` (but not for training itself)

| Coordinate system | Your files | YOLO `.txt` needs |
|-------------------|------------|-------------------|
| Map / Earth | GeoJSON/KML corners in meters or lon/lat | — |
| Pixels | Image grid | Normalized 0–1 polygon corners |

You **cannot** put lon/lat directly into `.txt`. You need a **georeferenced `.tif` per tile** to compute:

```
map corner → pixel corner → normalized YOLO .txt
```

After chipping:
- **Training** uses plain **PNG chips + `.txt`** only
- **GeoTIFF** is needed again for **DelAnyFlow inference** on full tiles

### Recommended labeling approach

1. Run **zero-shot DelAny** on georeferenced tiles → raw polygons
2. **Clean** in QGIS: fix misses, merges, false positives, boundaries
3. Use cleaned polygons as **ground truth** for fine-tuning

This is valid human-in-the-loop labeling (similar in spirit to FBIS manual correction).

---

## 3. Hardware Requirements

### Fine-tuning: Google Colab T4 GPU — recommended and sufficient

**Yes, you can use Google Colab with a T4 GPU** for DelAny fine-tuning on ~13k–20k chips.

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU | **T4 (16 GB VRAM)** | L4 / A100 (faster, not required) |
| Runtime | **GPU** | Do **not** select TPU |
| `batch` | 8 | 12 if no OOM; use 4 if OOM |
| `imgsz` | **512** | Must match DelAny |
| `epochs` | 15 | 15–20 |
| `lr0` | 1e-5 | Low LR for fine-tuning |
| RAM | 12 GB+ | Colab default is fine |
| Disk | ~5–15 GB | Google Drive for dataset + weights |
| Training time | ~1–4 hours | Depends on chip count and batch |

**Colab setup:** Runtime → Change runtime type → **T4 GPU**

### TPU — do not use

| | GPU (T4) | TPU |
|--|----------|-----|
| Ultralytics `model.train()` | **Supported** | **Not supported** in standard workflow |
| DelAny fine-tuning | **Works** | Will not work out of the box |

Ultralytics YOLO training targets **CUDA GPU**. TPU requires a different stack (XLA) and is not the right tool for this job.

### Colab tiers

| Tier | Notes |
|------|-------|
| Free | T4 sometimes; shorter sessions, may disconnect |
| Pro | More stable GPU access, longer runs |
| Pro+ | Faster GPUs; optional for this dataset size |

You do **not** need a local GPU or multi-GPU server for this scale.

### Data preparation (local machine)

| Tool | Purpose |
|------|---------|
| QGIS 3.x | Alignment QA, optional polygon fixes (not required for georeferencing) |
| Adobe Photoshop | Crop GEP export to match tile KML footprint |
| Python 3.10+ | Chipping script (`rasterio`, `geopandas`, `shapely`, `pillow`) |
| GDAL | Clip/export GeoTIFF (QGIS-bundled on Mac) |

**Mac GDAL path example:**

```bash
/Applications/QGIS.app/Contents/MacOS/gdalwarp
/Applications/QGIS.app/Contents/MacOS/gdal_translate
```

### What to upload to Google Drive for training

- `cambodia-fields/` folder (images + labels + `data.yaml`)
- `DelineateAnything.pt` (~125 MB)
- Save `best.pt` back to Drive when training completes

---

## 4. Project Overview & End-to-End Pipeline

### Assets you start with

- **32 tiles** (~25 km² each)
- **GEP PNG** exports per tile
- **KML** tile boundary + cleaned field polygons (GeoJSON/KML after DelAny + manual QA)
- **`DelineateAnything.pt`** base weights

### Pipeline (this project’s approach)

```
PHASE 1:   GEP export (PNG + tile KML) → crop in Adobe Photoshop to KML footprint
        ↓
PHASE 2:   Assign georeferencing from KML corners → GeoTIFF per tile (×32)
        ↓
PHASE 3:   QA — verify alignment on one pilot tile in QGIS, then lighter check per tile
        ↓
PHASE 4:   Chip 512×512 → PNG chips + YOLO seg .txt
        ↓
PHASE 5:   Organize dataset, split train/val by tile
        ↓
PHASE 6:   Fine-tune DelineateAnything.pt → best.pt (Colab T4)
        ↓
PHASE 7:   Run DelAnyFlow on GeoTIFFs with best.pt
```

This guide uses **GEP + Photoshop + KML-based georeferencing**, not QGIS Georeferencer. QGIS is still used later for **alignment QA** and optional polygon edits.

### Verify alignment once, then scale

Use **one pilot tile** (e.g. `bth_id_6`) to validate the full Phase 1–2 workflow in QGIS (pixel size ~1–2 m, extent ~5×5 km, polygons overlay fields). If it passes, apply the **same GEP export settings and Photoshop crop discipline** to all 32 tiles. Do a quick overlay check per tile before chipping.

---

## 5. Phase 1: Export from GEP & Crop in Photoshop

### Overview

For each of the 32 tiles:

1. Export high-resolution imagery from **Google Earth Pro (GEP)**
2. Save the matching **tile KML** (4-corner polygon from GEP “Save image”)
3. **Crop in Adobe Photoshop** so the image matches the KML tile footprint precisely
4. Pass the cropped image + KML to Phase 2 to build the GeoTIFF

```
GEP Save Image  →  tile_XX_gep.png + tile_XX.kml
        ↓
Photoshop crop  →  tile_XX_crop.png
        ↓
Phase 2         →  tile_XX_georeferenced.tif
```

### GEP export settings (keep identical across all 32 tiles)

| Setting | Recommendation |
|---------|----------------|
| Format | **PNG** preferred (or JPG) |
| Resolution | **Maximum** (~4800×4800 px cap) |
| Map options | **Off** (no labels, roads, borders overlay) |
| View | **Nadir** — tilt 0°, face north |
| Imagery date | Same or similar season where possible |
| Eye altitude | **Consistent** across tiles (record in a log) |

**Record per tile:** tile ID, eye alt, imagery date, export date, GEP version. Inconsistent exports make fine-tuning harder.

### Save image + KML together

When you click **Save image** in GEP:

1. Choose **Maximum** resolution and correct map options
2. Save the **image file** (PNG/JPG)
3. GEP also writes a **KML/KMZ** with a **4-corner ground overlay** describing the image footprint on Earth

Keep the image and KML as a **pair** for each tile:

```
tile_XX/
├── tile_XX_gep.png          # Raw GEP export (before crop)
├── tile_XX.kml              # GEP ground overlay / tile boundary (4 corners)
└── (later) tile_XX_fields.geojson   # Cleaned field polygons from DelAny + QA
```

The KML defines the **map coordinates** of the tile footprint. You will use these corners in Phase 2 to georeference the **cropped** image.

### Crop in Adobe Photoshop

**Goal:** the cropped image pixels should represent **exactly** the tile area defined by your KML polygon — no extra margin, no missing corners.

**Recommended steps:**

1. Open `tile_XX_gep.png` in Photoshop
2. Load or reference the tile KML footprint (overlay guide, traced mask, or measured crop box — use the same workflow you used for `bth_id_6`)
3. Crop the image to match the KML tile boundary as precisely as possible
4. Export as **PNG** (lossless) — e.g. `tile_XX_crop.png`
5. **Do not resize** after crop unless you intentionally want to change GSD (avoid accidental scale changes)

**Important rules after Photoshop crop:**

| Rule | Why |
|------|-----|
| Crop only — avoid stretch/skew | Stretch breaks the simple corner-to-corner georef math |
| Keep RGB (no extra effects) | DelAny expects natural RGB |
| Note final **width × height in pixels** | Needed for Phase 2 geotransform |
| Use the **same KML** that defines the tile footprint | Corners map to cropped image edges |

**Known risk:** Photoshop changes pixel dimensions without updating map coordinates automatically. Phase 2 must assign georef to the **cropped** image dimensions, not the original GEP export size.

### Files after Phase 1 (per tile)

```
tile_XX_crop.png     # Cropped RGB image ready for georeferencing
tile_XX.kml          # Tile boundary with map coordinates (4 corners)
```

---

## 6. Phase 2: Create GeoTIFF from KML + Cropped Image

### Overview

Take the **Photoshop-cropped PNG** and the **tile KML**, then build a **georeferenced GeoTIFF** by linking:

- **KML corner coordinates** (map / Earth position)
- **Cropped image corners** (pixel position: top-left, top-right, bottom-right, bottom-left)

This is a **corner-pin georeference**: the full image is assumed to linearly span the KML rectangle (same assumption GEP’s ground overlay uses).

```
tile_XX_crop.png  +  tile_XX.kml (4 corners in map coords)
        ↓
Assign geotransform / write GeoTIFF
        ↓
tile_XX_georeferenced.tif
```

### Step 1 — Get KML corner coordinates in projected CRS

GEP KML is usually in **WGS84 (EPSG:4326)** lon/lat. For Cambodia, convert corners to a **projected CRS in meters**, e.g. **EPSG:32648** (UTM zone 48N).

From the 4 corners, compute the tile bounding box in UTM:

| Corner | Role |
|--------|------|
| Top-left | Upper-left map coordinate |
| Top-right | Upper-right |
| Bottom-right | Lower-right |
| Bottom-left | Lower-left |

Derive:

- `xmin`, `xmax` (easting)
- `ymin`, `ymax` (northing)

**Example** (from a pilot tile like `bth_id_6`):

- Extent ~**5000 m × 5000 m** (25 km²)
- Pixel size ~**1.0–1.1 m** for a ~4750×4790 px crop

### Step 2 — Assign georeferencing to the cropped image

The geotransform links map coordinates to pixels:

```
map_x = origin_x + column × pixel_width
map_y = origin_y + row   × pixel_height    (pixel_height usually negative)
```

Where:

- `origin_x`, `origin_y` = map coordinate of the **top-left pixel** (usually xmin, ymax)
- `pixel_width` = (xmax − xmin) / image_width
- `pixel_height` = (ymin − ymax) / image_height  ← negative for north-up images

You can build the GeoTIFF with **GDAL**, a small script, or any tool you already use — as long as the output is a valid GeoTIFF with embedded CRS and geotransform.

### Option A — GDAL `gdal_translate` (corner bounding box)

After converting KML corners to UTM (EPSG:32648), if the tile is a **north-up rectangle**:

```bash
export PROJ_LIB="/Applications/QGIS.app/Contents/Resources/proj"
export PROJ_DATA="/Applications/QGIS.app/Contents/Resources/proj"

"/Applications/QGIS.app/Contents/MacOS/gdal_translate" \
  -of GTiff \
  -a_srs EPSG:32648 \
  -a_ullr <xmin> <ymax> <xmax> <ymin> \
  "/path/to/tile_XX_crop.png" \
  "/path/to/tile_XX_georeferenced.tif"
```

Replace `<xmin> <ymax> <xmax> <ymin>` with UTM coordinates from the KML corners.

`-a_ullr` sets: upper-left X, upper-left Y, lower-right X, lower-right Y — matching a north-up image where top row = ymax and bottom row = ymin.

### Option B — Your existing manual workflow

If you already have a trusted process that produced tiles like `bth_id_6_image_crop_georeferenced.tif` (~1.05 m, EPSG:32648, ~5×5 km), **keep using that same method** for all 32 tiles. The requirements are:

1. Input = **cropped** image from Phase 1 (not raw GEP export)
2. Corners = **KML map coordinates** matched to image edges
3. Output = **GeoTIFF** with CRS in meters (UTM)

### Step 3 — Validate the GeoTIFF in QGIS

Open `tile_XX_georeferenced.tif` and check **Layer Properties → Information**:

| Check | Target |
|-------|--------|
| CRS | EPSG:32648 (or your chosen UTM zone) |
| Pixel size X / Y | ~**1–2 m** (absolute value of Y) |
| Width × Height | Matches your cropped PNG dimensions |
| Extent span | ~**5000 m × 5000 m** for a 25 km² tile |

**Pilot reference** (`bth_id_6_image_crop_georeferenced.tif`):

- Pixel size: ~**1.05 m**
- Dimensions: ~**4759 × 4789** px
- Extent: ~**5 km × 5 km**

### Step 4 — RGB-only export (if needed)

If your output has 4 bands (RGB + alpha) or you used JPG:

```bash
"/Applications/QGIS.app/Contents/MacOS/gdal_translate" \
  -b 1 -b 2 -b 3 \
  "/path/to/tile_XX_georeferenced.tif" \
  "/path/to/tile_XX_rgb.tif"
```

For DelAny training prep and inference, **3-band RGB** is sufficient.

### Final deliverable per tile

```
tile_XX_georeferenced.tif   # Georeferenced RGB GeoTIFF (~1–2 m, UTM)
tile_XX_fields.geojson      # Cleaned field polygons (same CRS as TIF)
```

**You need 32 georeferenced `.tif` files** (one per tile) to build `.txt` labels and to run DelAnyFlow after fine-tuning.

### Accuracy notes for this workflow

| Risk | Mitigation |
|------|------------|
| Photoshop crop does not match KML footprint | Crop carefully; verify extent ≈ 5×5 km in QGIS |
| Stretch/skew in Photoshop | Crop only; no transform |
| KML from original export used with different crop size | Always georef the **cropped** image; recompute pixel size from new dimensions |
| GEP tilt / terrain | Keep nadir view; consistent eye alt |
| Polygon misalignment | Phase 3 QA — overlay cleaned polygons on TIF before chipping |

---

## 7. Phase 3: Alignment QA

**Do not skip this on your pilot tile.**

### QGIS visual check

1. Add `tile_XX_georeferenced.tif`
2. Add `tile_XX_fields.geojson` (or `.kml`)
3. Confirm polygons sit **exactly** on field boundaries

| Result | Action |
|--------|--------|
| Aligned | Proceed to chipping |
| Shifted / scaled / rotated | Re-check KML corners vs cropped image; redo Photoshop crop or Phase 2 georef |
| Polygons from a different image version | Re-run DelAny on current TIF or re-export polygons aligned to this TIF |

### Polygon quality

Each training instance = **one field**:

- One closed polygon per field
- Split merged multi-fields
- Delete false positives
- Add missed fields
- Fix obvious boundary errors along roads/trees/water

### Pilot first

Complete Phases 1–4 on **one tile** before batching all 32.

---

## 8. Phase 4: Chip Imagery & Build YOLO Labels

### Chipping parameters

| Parameter | Value |
|-----------|--------|
| Chip size | **512 × 512** pixels |
| Stride | **256** (50% overlap) |
| Output image format | **PNG** (plain, no georef on chips) |
| Label format | **YOLO instance segmentation** |
| Class | `0` = `field` |

### Expected chip count

For ~5 km × 5 km at ~1 m GSD (~5000×5000 px):

- Roughly **400–625 chips per tile**
- **32 tiles → ~13,000–20,000 chips** total

### YOLO segmentation label format

One `.txt` per chip, **same basename** as the image.

**One line per field:**

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

- `class_id`: **0**
- `x`, `y`: normalized to **[0, 1]** relative to chip width/height

**Example** (two fields in one chip):

```
0 0.12 0.34 0.15 0.38 0.22 0.35 0.18 0.31
0 0.55 0.60 0.58 0.62 0.61 0.58 0.57 0.55
```

**Empty chip:** empty `.txt` file (valid negative example).

**Do not use detection-only bbox format** (`0 cx cy w h`) — DelAny needs polygon segmentation labels.

### Map → pixel conversion

For each polygon vertex in GeoJSON (map coordinates):

```
1. Map (easting, northing)
        ↓  inverse geotransform from .tif
2. Pixel (col, row) in full tile image
        ↓  subtract chip origin (col0, row0)
3. Pixel (px, py) in 512×512 chip
        ↓  divide by 512
4. Normalized (xn, yn) for YOLO .txt
```

### Chipping algorithm (per tile)

```
FOR each 512×512 window with stride 256 on tile_XX_georeferenced.tif:
    1. Read RGB window → save as tile_XX_chipNNNN.png
    2. Get window bounds in map coordinates
    3. Clip field polygons to window
    4. FOR each clipped polygon:
           - Convert vertices map → chip pixels
           - Normalize to 0–1
           - Write one line to .txt
    5. If no polygons: write empty .txt
```

### Implementation

| Method | Best for |
|--------|----------|
| **Python** (`rasterio` + `geopandas`) | All 32 tiles — recommended |
| QGIS Processing | 1–2 tiles only |

### Visual QA after chipping

Spot-check **≥20 random chips**: overlay polygon on PNG. Do not train until the pilot tile passes.

---

## 9. Phase 5: Dataset Layout & Train/Val Split

### Folder structure

```
cambodia-fields/
├── images/
│   ├── train/
│   │   ├── tile01_chip0000.png
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── tile01_chip0000.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

**Rule:** `labels/{split}/{name}.txt` must pair with `images/{split}/{name}.png` (same basename).

### Train / validation split — by tile

| Split | Tiles | Notes |
|-------|-------|-------|
| **Train** | ~25 tiles | All chips → `train/` |
| **Val** | ~7 tiles | All chips → `val/` |

**Never** put chips from the same tile in both train and val.

Example split record:

```csv
tile_id,split
bth_id_01,train
bth_id_28,val
```

### `data.yaml`

```yaml
path: /path/to/cambodia-fields
train: images/train
val: images/val

nc: 1
names: ['field']
```

---

## 10. Phase 6: Transfer Learning (Fine-Tuning)

### Install (Colab)

```python
!pip install ultralytics==8.3.148
```

Download [`DelineateAnything.pt`](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt).

### Mount Drive & verify

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify: image count == label count in train and val
```

### Training script

```python
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/models/DelineateAnything.pt")

results = model.train(
    data="/content/drive/MyDrive/cambodia-fields/data.yaml",
    task="segment",
    imgsz=512,
    epochs=15,
    batch=8,              # try 12 on T4; use 4 if OOM
    lr0=1e-5,             # low LR for fine-tuning
    lrf=0.01,
    patience=5,
    mosaic=1.0,
    project="/content/drive/MyDrive/runs",
    name="cambodia-delany-v1",
    exist_ok=True,
)
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `imgsz` | 512 | Must match DelAny |
| `epochs` | 15–20 | Start with 15 |
| `batch` | 8 | Reduce to 4 if OOM |
| `lr0` | 1e-5 | Fine-tune, not from scratch |
| `task` | segment | Instance segmentation |
| Base weights | `DelineateAnything.pt` | Not raw COCO weights |

### Outputs

```
runs/segment/cambodia-delany-v1/weights/
├── best.pt    ← use for inference
└── last.pt
```

### Metrics to watch

| Metric | Expectation |
|--------|-------------|
| `train/seg_loss` | Should decrease |
| `val/seg_loss` | Should decrease |
| `metrics/mAP50(M)` | Segmentation mAP — rough quality indicator |

If train loss decreases but val loss increases → overfitting (fewer epochs, more data, or check label quality).

### Smoke test before full run

1. Chip **one tile** (~500 chips)
2. Train **5 epochs** on Colab T4
3. Confirm loss decreases and no errors
4. Batch all 32 tiles → full 15–20 epoch run

---

## 11. Phase 7: Inference with Fine-Tuned Weights

Training uses **PNG chips**. Production inference uses **GeoTIFF** + **DelAnyFlow** (`delineate.py`).

### Update config

In `conf_sample.yaml` or your batch config:

```yaml
model: ["/path/to/best.pt"]
```

### Band order for GEP RGB GeoTIFF

```yaml
data_loader:
  bands: [1, 2, 3]   # RGB — not [3,2,1] used for Sentinel-2
```

### Run inference

```bash
python delineate.py -b batch_sample.yaml
```

Output: georeferenced field polygons in `data/delineated/` (GPKG).

### Compare results

- Zero-shot `DelineateAnything.pt` vs fine-tuned `best.pt`
- Visual review on **val tiles** (not used in training)
- Compare against your cleaned polygons

---

## 12. Troubleshooting

| Problem | Cause | Fix |
|---------|--------|-----|
| Labels shifted on chips | KML corners don’t match cropped image; wrong CRS | Re-do Phase 1–2; re-check alignment in QGIS |
| Cannot build `.txt` from GeoJSON | Map coords without image geotransform | Need georeferenced `.tif` per tile |
| Model worse than zero-shot | Bad cleanup propagated | Re-QA polygons; add missed fields |
| Val metrics too optimistic | Same tile in train & val | Split by tile |
| CUDA OOM on Colab | Batch too large | `batch=4` or `batch=2` |
| Wrong colors at inference | Wrong band order | `bands: [1,2,3]` for RGB GeoTIFF |
| QGIS `proj.db` error on Mac | PROJ path not set | Set `PROJ_LIB` / `PROJ_DATA` (see Phase 2) |

---

## 13. Checklists

### Pilot tile (do first)

- [ ] GEP PNG + KML exported with consistent settings (max res, nadir, map options off)
- [ ] Photoshop crop matches KML tile footprint → `tile_XX_crop.png`
- [ ] GeoTIFF built from KML corners → `tile_XX_georeferenced.tif`
- [ ] QGIS check: CRS UTM, pixel size ~1–2 m, extent ~5×5 km
- [ ] Cleaned field polygons in same CRS
- [ ] QGIS alignment QA passed (polygons on field boundaries)
- [ ] Chips + `.txt` generated
- [ ] ≥5–20 chips visually verified
- [ ] 5-epoch smoke test on Colab T4 passed

### All 32 tiles

- [ ] Same GEP + Photoshop + KML→TIF workflow applied to each tile
- [ ] Quick overlay check per tile before chipping
- [ ] All chips 512×512 PNG
- [ ] All labels YOLO seg polygons
- [ ] Image/label basenames match
- [ ] Train/val split by tile (~25 / ~7)
- [ ] `data.yaml` created
- [ ] Dataset uploaded to Google Drive

### Training & inference

- [ ] `DelineateAnything.pt` on Drive
- [ ] Ultralytics ~8.3.148 installed
- [ ] Full train 15–20 epochs on Colab T4
- [ ] `best.pt` saved
- [ ] DelAnyFlow run with `best.pt` on GeoTIFFs
- [ ] Compared zero-shot vs fine-tuned on val tiles

---

## Quick Reference

```
GEP PNG + KML  →  Photoshop crop  →  KML georef  →  .tif (×32)
       +
Cleaned GeoJSON / KML (map coords)
       ↓
Chip 512×512 + map→pixel  →  PNG chips + .txt
       ↓
data.yaml + split by tile
       ↓
YOLO(DelineateAnything.pt).train() on Colab T4  →  best.pt
       ↓
delineate.py on .tif with best.pt  →  GPKG
```

| Stage | GeoTIFF needed? | PNG + .txt needed? |
|-------|-----------------|---------------------|
| Build labels | **Yes** | Output of this step |
| Fine-tuning | No | **Yes** |
| DelAnyFlow inference | **Yes** | No |

---

## References

- [Delineate Anything paper](https://arxiv.org/abs/2504.02534)
- [DelAnyFlow paper](https://arxiv.org/abs/2511.13417)
- [DelineateAnything.pt on HuggingFace](https://huggingface.co/MykolaL/DelineateAnything)
- [FBIS-22M dataset](https://huggingface.co/datasets/MykolaL/FBIS-22M)
- [Ultralytics YOLO docs](https://docs.ultralytics.com/)
- [Colab demo](https://colab.research.google.com/drive/10KSLwYDTgU-WhpqqG39yyvB6K8MdB0X9)
- See also: `delineation_config_guide.md` for DelAnyFlow inference configuration
