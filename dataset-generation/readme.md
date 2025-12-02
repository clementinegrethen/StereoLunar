# Lunar Surface Dataset Generator

This repository provides scripts to generate synthetic stereo image pairs of the lunar surface using the [SurRender](https://www.airbus.com/en/products-services/space/space-equipment/surrender) rendering engine.

## Overview

The pipeline consists of three main steps:

1. **Trajectory Generation** - Create camera and sun trajectories (CSV files)
2. **Image Rendering** - Render images and metadata using SurRender
3. **Dataset Processing** - Convert outputs to a standardized format compatible with MASt3R

---

## Data Files (`datas/` folder)

This repository includes some essential data files for SurRender (you still need to compute .big heightmap & conemap!):

```
datas/
├── DEM/
│   └── south5m.dem                    # LOLA South Pole DEM configuration
└── texture/
    ├── ldem_875s_5m_conemap.geo       # Cone map texture reference
    └── ldem_875s_5m_heightmap.geo     # Height map texture reference
```

### LOLA DEM Setup

The scripts use the LOLA South Pole 5m DEM from NASA:

1. **Download** the DEM from: https://pgda.gsfc.nasa.gov/products/78
2. **Convert** to SurRender format following the user documentation:
   - Generate `.big` files for heightmap and conemap
   - Generate `.geo` reference files
3. **Place files** in SurRender's resource path:
   - `textures/`: `.geo` and `.big` files
   - `dem/`: `.dem` configuration file
4. **Hapke BRDF**: Already available in SurRender's `materials/` folder

---

## Requirements

### Software
- [SurRender](https://www.airbus.com/en/products-services/space/space-equipment/surrender) rendering engine (with Python client)
- Python 3.8+

### Python Dependencies
```bash
pip install numpy pandas scipy opencv-python imageio tqdm
```

### descentimagegenerator Library

Install the provided `descentimagegenerator` library:
```bash
cd descentimagegenerator
pip install -e .
```

---

## Pipeline

### Step 1: Generate Trajectories

Choose a trajectory type based on your stereo configuration. **Edit the script** to configure parameters, then run:

#### Nadir Stereo Pairs
Cameras pointing straight down (perpendicular to surface).

```bash
python nadir_trajectory.py
```

**Key parameters to edit in script:**
- `ALTITUDE_DISTRIBUTION`: altitude levels and number of pairs per level
- `B_H_RATIO_MIN/MAX`: baseline-to-height ratio range (0.02 - 0.10)
- `LAT_RANGE`, `LON_RANGE`: geographic coverage area
- `s.connectToServer("127.0.0.1")`: SurRender server address
- `s.createSphericalDEM(...)`: path to your lunar DEM

**Output files (in `output/` folder):**
- `camera_trajectory.csv` - Camera positions (x, y, z) and orientations (q0, qx, qy, qz)
- `sun_trajectory.csv` - Sun positions for lighting
- `metadata.csv` - Detailed metadata per image
- `pair_summary.csv` - Summary statistics per pair

---

#### Oblique Stereo Pairs
Cameras viewing at an angle (26°-48° from horizontal).

```bash
python oblique_trajectory.py
```

**Key parameters to edit in script:**
- `DISTANCE_TO_ALTITUDE_RATIO_MIN/MAX`: controls viewing angle (0.5 - 1.1)
- `ALTITUDE_STRATEGIES`: ["same", "close", "different", "progressive"]
- `STEREO_BASELINE`: lateral separation between cameras (25m default)

**Output files:**
- `camera_trajectory_oblique.csv`
- `sun_trajectory_oblique.csv`
- `metadata_oblique.csv`
- `pair_summary_oblique.csv`

---

#### Dynamic Stereo Pairs
Cameras with varied, asymmetric viewing geometries (challenging pairs).

```bash
python dynamic_trajectory.py
```

**Key parameters to edit in script:**
- `CAMERA_MODES`: ["mild_dynamic", "moderate_dynamic", "extreme_dynamic"]
- `ALT_VARIATION_RANGE`: altitude variation per camera (±30%)
- `ROLL_VARIATION_RANGE`: roll variation (±10°)
- `B_H_RATIO_MIN/MAX`: baseline ratio (0.02 - 0.22)

**Output files:**
- `camera_trajectory_landing.csv`
- `sun_trajectory_landing.csv`
- `metadata_landing.csv`
- `pair_summary_landing.csv`

---

### Step 2: Render Images

Use the generated trajectories to render images with SurRender.

**First**, edit `generate_images.py` to configure your scenario in `get_scenarii_dict()`:
```python
SCENARII = dict(
    MR=(
        SceneConfig(
            DEM_path="path/to/your/lunar.dem",
            body_brdf="hapke.brdf",
        ),
        TrajectoryConfig(
            trajectory_path="output/camera_trajectory.csv",
            sun_trajectory_path="output/sun_trajectory.csv",
        ),
    ),
)
```

**Then run:**
```bash
python generate_images.py MR
```

**Command-line options:**

| Argument | Description | Default |
|----------|-------------|---------|
| `MR` | Scenario name (required) | - |
| `--host` | SurRender server hostname | `localhost` |
| `--port` | SurRender server port | `5151` |
| `--timeout` | Connection timeout (seconds) | `120.0` |
| `--image_rays` | Rays per pixel (quality) | `64` |
| `--image_step` | Render every N frames | `1` |
| `--metadata_step` | Generate metadata every N frames | `1` |
| `--skip_image` | Skip image rendering | `False` |
| `--skip_metadata` | Skip metadata generation | `False` |

**Examples:**
```bash
# Full rendering with high quality
python generate_images.py MR --host localhost --port 5151 --image_rays 128

# Quick preview (every 10th frame)
python generate_images.py MR --image_step 10 --metadata_step 10

# Metadata only (no images)
python generate_images.py MR --skip_image

# Images only (no metadata)
python generate_images.py MR --skip_metadata
```

**Output structure:**
```
outputs/MR/
├── images/
│   ├── im_00000.png
│   ├── im_00001.png
│   └── ...
└── metadata/
    ├── im_00000.npz
    ├── im_00001.npz
    └── ...
```

**Metadata NPZ contents:**
- `dmap`: Depth map (distance from camera to surface)
- `los_map`: Line-of-sight direction vectors
- `label_map`: Semantic segmentation (255=surface, 0=background)
- `K`: 3x3 camera intrinsic matrix
- `R_w2c`: 3x3 world-to-camera rotation matrix
- `t_w2c`: 3x1 world-to-camera translation vector

---

### Step 3: Process Dataset

Convert SurRender outputs to standardized format for MASt3R training/evaluation.

```bash
python process_dataset.py --images_dir ./outputs/MR/images \
                          --metadata_dir ./outputs/MR/metadata \
                          --output_dir ./dataset/lunar_stereo
```

**Command-line arguments:**

| Argument | Description | Required |
|----------|-------------|----------|
| `--images_dir` | Directory containing PNG/JPG images | Yes |
| `--metadata_dir` | Directory containing NPZ metadata files | Yes |
| `--output_dir` | Output directory for processed dataset | Yes |

**Output format:**
```
dataset/lunar_stereo/
├── im_00000.jpg      # Compressed image
├── im_00000.exr      # Depth map (Z-buffer, float32)
├── im_00000.npz      # Camera parameters
├── im_00001.jpg
├── im_00001.exr
├── im_00001.npz
└── ...
```

**Output NPZ contents:**
- `intrinsics`: 3x3 camera intrinsic matrix K
- `cam2world`: 4x4 camera-to-world transformation matrix

---

## Trajectory Types Comparison

| Type | Viewing Angle | Camera Symmetry | Use Case |
|------|---------------|-----------------|----------|
| **Nadir** | 0° (vertical) | Symmetric | Classical stereo |
| **Oblique** | 26°-48° | Symmetric | Terrain features |
| **Dynamic** | Variable | Asymmetric | Robust algorithms |

---

## Configuration Reference

### Altitude Distribution
All trajectory scripts use configurable altitude levels:

```python
ALTITUDE_DISTRIBUTION = [
    {"altitude": 3500,  "pairs": 10},
    {"altitude": 6500,  "pairs": 10},
    {"altitude": 9500,  "pairs": 10},
    # ... up to 30500m
]
```

### Baseline Ratio (B/H)
The baseline-to-height ratio controls stereo geometry:
- **Nadir**: `B_H_RATIO_MIN = 0.02`, `B_H_RATIO_MAX = 0.10`
- **Dynamic**: `B_H_RATIO_MIN = 0.02`, `B_H_RATIO_MAX = 0.22`

### Lighting Configuration
Three sun positions are used per pair:
```python
SUN_SETUPS = [
    (150, 160),  # Azimuth: 150°, Incidence: 160°
    (250, 20),   # Azimuth: 250°, Incidence: 20°
    (360, 165)   # Azimuth: 360°, Incidence: 165°
]
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{Grethen_2025_ICCV,
    author    = {Grethen, Cl\'ementine and Gasparini, Simone and Morin, G\'eraldine and Lebreton, J\'er\'emy and Marti, Lucas and Sanchez-Gestido, Manuel},
    title     = {Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {3751-3760}
}
```

---

## Author

**Clémentine GRETHEN**

## License

[Your License Here]
