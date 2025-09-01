# MOONSt3R  
**Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset**  
*Accepted at ICCV 2025 (3D-VAST Workshop)*  

---

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge"></a>
  <a href="https://clementinegrethen.github.io/publications/3D-Vast-ICCV2025.html"><img src="https://img.shields.io/badge/Project%20Page-Online-blue?style=for-the-badge"></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-StereoLunar-green?style=for-the-badge"></a>
</p>

---
## 📂 Dataset Description

Our dataset is provided in a format directly compatible with **DUSt3R/MASt3R training pipelines**.  
For each image, we include three synchronized files:

- **RGB image (`.jpg`)**  
  - Resolution: **512 × 512** pixels  
  - Standard 8-bit image for visualization and training.  
  - All images are expressed in a **selenocentric (Moon-centered) coordinate system**.

- **Depth map (`.exr`)**  
  - High-precision floating-point depth map (in meters).  
  - Encodes the distance from the camera to the visible surface for each pixel.  
  - Can be used to generate ground-truth 3D point clouds.

- **Camera metadata (`.npz`)**  
  - Contains the intrinsic and extrinsic calibration matrices:  
    - `intrinsics` (or `K`): 3×3 camera intrinsic matrix  
    - `extri` or `Rt`: 3×4 camera pose matrix (world→camera, OpenCV convention)  
    - Alternative fields (`fx`, `fy`, `cx`, `cy`) are also provided for compatibility.  
  - This ensures reproducible reprojections and geometric consistency.

---

## 📁 Data Structure

A typical folder looks like this:
dataset_root/
│
├── im_00000.jpg
├── im_00000.exr
├── im_00000.npz
│
├── im_00001.jpg
├── im_00001.exr
├── im_00001.npz
│
└── ...

Each triplet `{.jpg, .exr, .npz}` corresponds to a single camera frame.  

---


- In the repository, the folder **`DataGeneration/`** contains a **small example subset** of the dataset.  
  This is meant for quick testing.

- The dataset has been formatted to be **plug-and-play with DUSt3R/MASt3R**.  
  We also provide (📌 *TODO*) a **custom dataloader** that loads images, depth maps, and camera parameters in one call.

---

## 📄 Citation
If you find this work useful, please cite:

> Grethen, C., Morin, G., Gasparini, S., Lebreton, L., Marti, J., & Gestido, M. S. (2025, October).  
> *Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset.*  
> Accepted in **ICCV Workshops (3D-VAST 2025)**.

```bibtex
@inproceedings{grethen2025moonst3r,
  title={Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset},
  author={Grethen, Clémentine and Morin, Géraldine and Gasparini, Simone and Lebreton, Jérémy and Marti, Lucas and Gestido, Manuel Sanchez},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2025},
  month={October},
  note={Accepted at 3D-VAST Workshop}
}
```

## Acknowledgements

This work was supported by the **European Space Agency (ESA)** under contract  
**4000140461/23/NL/GLC/my**.  

We gratefully acknowledge the valuable support and collaboration of the  
**European Space Agency (ESA)** and **Airbus Defence and Space**.
