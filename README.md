# StereoLunar ICCV 3D-VAST 2025 
Official implementation of **Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset**
*Accepted at ICCV 2025 (3D-VAST Workshop)*

<p align="center">
  <img src="assets/illustration3DR.png" alt="3D Reconstruction Illustration" width="800"/>
</p>
---

<p align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2025W/3D-VAST/html/Grethen_Adapting_Stereo_Vision_From_Objects_To_3D_Lunar_Surface_Reconstruction_ICCVW_2025_paper.html">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge">
  </a>
  <a href="https://clementinegrethen.github.io/publications/3D-Vast-ICCV2025.html">
    <img src="https://img.shields.io/badge/Project%20Page-Online-blue?style=for-the-badge">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Dataset-StereoLunar-green?style=for-the-badge">
  </a>
</p>

---


## 📋 Table of Contents

- [News](#-news)
- [Installation](#-installation)
- [Checkpoints](#-checkpoints)
- [Usage: Inference & Demo](#-usage-inference--demo)
  - [Sample Data](#sample-data)
  - [Demo Scripts](#demo-scripts)
  - [Advanced Usage: Feature Matching](#-advanced-usage-feature-matching)
- [Dataset Description](#dataset-description)
- [Data Structure](#data-structure)
- [Finetuning Example](#-finetuning-example)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🚀 News
**Coming Soon:** New model for 3D Moon reconstruction based on VGGT (CVPR 2025) and StereoLunar dataset!

---

## 📦 Installation

### Prerequisites
- Python 3.11
- CUDA-compatible GPU (recommended)
- Git with submodule support

### Step-by-step Installation

**1. Clone the repository**
```bash
git clone --recursive https://github.com/clementinegrethen/StereoLunar.git
cd StereoLunar

# If you have already cloned without submodules:
# git submodule update --init --recursive
```

**2. Create and activate the environment**
```bash
conda create -n StereoLunar python=3.11 cmake=3.14.0
conda activate StereoLunar 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add required packages for visloc.py
pip install -r dust3r/requirements_optional.txt
```

**3. Compile and install ASMK**
```bash
pip install cython
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .
cd ..
```

**4. (Optional) Compile CUDA kernels for RoPE acceleration**
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

---


##  Checkpoints

There are two options for downloading our StereoLunar fine-tuned MASt3R model:

**Option 1: Automatic download via Hugging Face Hub**
The model ([StereoLunar](https://huggingface.co/cgrethen/StereoLunar)) will be downloaded automatically when you run the demo.

**Option 2: Manual download**
```bash
# Install gdown if needed
pip install gdown

# Create checkpoints directory and download model
mkdir -p checkpoints/
gdown --fuzzy "https://drive.google.com/file/d/11PjhqADOOXfIkLk64ognltVN-gUHxLHA/view?usp=drive_link" -O checkpoints/  # StereoLunar.pth
```

##  Usage: Inference & Demo

### Sample Data
A folder with sample scenes from our **StereoLunar dataset** (see [🌙 Dataset Description](#dataset-description)) for quick testing is provided: [`quick_testing/`](quick_testing/) *. We provided 3 pairs for each type of trajectories: Nadir, Oblique and Dynamic.

### Demo Scripts

**Run interactive demo:**
```python
python demo.py --weights checkpoints/StereoLunar.pth
```

**Run non-interactive demo:**
```python
python demo_mast3r_nongradio.py --weights checkpoints/StereoLunar.pth
```
## Advanced Usage: 
### Feature Matching
<details>
<summary><strong>Click to expand:</strong> Code sample to compute matches with StereoLunar for a pair of images</summary>

```python
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    device = 'cuda'
    model_name = "checkpoints/StereoLunar.pth"  # Updated path
    # You can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # Load your images - replace with actual paths
    images = load_images(['path/to/img1.jpg', 'path/to/img2.jpg'], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # At this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy() 
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy() 

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()
    
    # Find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # Ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # Visualize matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl

    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    
    pl.figure(figsize=(15, 8))
    pl.imshow(img)
    pl.title('Feature Matches between Lunar Surface Images')
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.axis('off')
    pl.tight_layout()
    pl.show(block=True)
```
---
</details>

**Example Result:**

<p align="center">
  <img src="assets/image.png" alt="Feature Matching Example" width="500"/>
</p>

---
### Save PLY of the reconstructed scene
<details>
<summary><strong>Click to expand:</strong> Code sample to save 3D reconstruction scene with StereoLunar for a pair of images</summary>

```python
import open3d as o3d
import numpy as np
import cv2
from PIL import Image

def make_pointcloud(pts3d, rgb_img):
    if pts3d.shape[:2] != rgb_img.shape[:2]:
        print(f"[!] Resize image from {rgb_img.shape[:2]} to match 3D shape {pts3d.shape[:2]}")
        rgb_img = cv2.resize(rgb_img, (pts3d.shape[1], pts3d.shape[0]))

    mask = np.isfinite(pts3d).all(axis=2)
    pts = pts3d[mask].reshape(-1, 3)
    colors = rgb_img[mask].reshape(-1, 3) / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def preprocess_image_as_inference(img_path, size=512, square_ok=False):
    img = Image.open(img_path).convert('RGB')
    W1, H1 = img.size
    img = img.resize((size, size), resample=Image.BICUBIC)

    cx, cy = size // 2, size // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if not square_ok and W1 == H1:
        halfh = int(3 * halfw / 4)
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return np.array(img)

rgb0_crop = preprocess_image_as_inference(image_path1)
rgb1_crop = preprocess_image_as_inference(image_path2)
# pcd0 en rouge, pcd1 en bleu
pcd0 = make_pointcloud(pts3d_im0, rgb0_crop)
pcd1 = make_pointcloud(pts3d_im1, rgb1_crop)

o3d.visualization.draw_geometries([pcd0, pcd1])
pcd_combined = pcd0 + pcd1

o3d.io.write_point_cloud("pcd_combined_from_imgs.ply", pcd_combined, write_ascii=False)

print("Save: 'pcd_combined_from_imgs.ply'")
```
---
</details>

## 🌙 SteroLunar Dataset 
⚠️ IMPORTANT
The data may be updated over time, and additions can be made.  
Below is the **record of changes**:

### Record of Changes
- **2025-10-15** — Initial release of the dataset documentation.  


### Full Dataset access 
We are hosting the datas on Zenodo: 
### Dataset Description

<p align="center">
  <img src="assets/StereoLunar.png" alt="StereoLunar Dataset" width="600"/>
</p>

Our dataset is provided in a format directly compatible with **DUSt3R/MASt3R training pipelines**.  

On the dataset Zenodo page, you can find 3 types of trajectories:

-Nadire
-Oblique
-Dynamic 
And several lighting conditions for each pair as well as many altitudes of camera. (Please refer to the research paper for more informatiion)
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


## Data Structure

A typical folder looks like this:

```
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
```

Each triplet `{.jpg, .exr, .npz}` corresponds to a single camera frame.

---
## Integration of StereoLunar with StereoLunar:
The dataset has been formatted to be **plug-and-play with DUSt3R/MASt3R**. We also provide  a **custom dataloader** that loads images, depth maps, and camera parameters in one call.  [`mast3r/dust3r/dust3r/datasets/LunarDataset.py`](mast3r/dust3r/dust3r/datasets/LunarDataset.py)

---

### DataLoader for LunarStereo

To train with the StereoLunar dataset, we provide a dataloader in [`mast3r/dust3r/dust3r/datasets/`](mast3r/dust3r/dust3r/datasets/) directly compatible with the format above. No preprocessing required.

Images are loaded as 512x512 `.jpg` files. Each view is paired with:
- a `.npz` file containing camera intrinsics and extrinsics
- a `.exr` file containing a depth map in metric scale
We also add some Moon-specific data augmentation in [`mast3r/mast3r/datasets/base/mast3r_base_stereo_view_dataset.py`](mast3r/mast3r/datasets/base/mast3r_base_stereo_view_dataset.py)
---



## Reproduce fine-tuning procedure: Finetuning Example
### MAST3R checkpoint:
MASt3R Model
You can obtain the model checkpoints by two ways:

You can use the huggingface_hub integration: the models will be downloaded automatically.

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | CatMLP+DPT | ViT-L | ViT-B |
### Fine-tune:

Otherwise, download it from our server:
We recommend not forcing the fine-tuning to be strictly metric, as the Moon’s altitude is particularly difficult to learn due to its fractal-like surface patterns.

The split file defines which pairs are used for training and testing. For effective fine-tuning, it is important to ensure a sufficient mix of pairs covering diverse trajectories, altitudes, and illumination conditions.

To finetune on your own StereoLunar dataset (you can use torchrun if you have several gpus):

```python

# !export CUDA_VISIBLE_DEVICES=0  # UNE SEULE GPU pour éviter overhead torchrun
# !CUDA_VISIBLE_DEVICES=1
!export CUDA_VISIBLE_DEVICES=0,1
!export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

!python3 train.py \
--train_dataset="LunarDataset(split='train',ROOT='../',split_file='split.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto',aug_monocular=0.005,use_lunar_augmentations=True,lunar_grayscale_tint_prob=0.3,lunar_grayscale_alpha_range=[0.3, 1.0],lunar_bilateral_filter_prob=0.5,lunar_bilateral_params=[15, 75, 100],lunar_contrast_variation_prob=0.7,lunar_contrast_factors=[0.5, 1.0, 1.5])" \
--test_dataset="LunarDataset(split='test',ROOT='../',split_file='split.npz',resolution=512,n_corres=8192,nneg=0.5)" \
--model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf),freeze='encoder')" \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
--test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=False, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
--pretrained "checkpoint/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
--lr 3e-5 --min_lr 2e-07 --warmup_epochs 0 --epochs 25 --batch_size x --accum_iter 1 \
--save_freq 1 --keep_freq 3 --eval_freq 2 --print_freq=20 --disable_cudnn_benchmark \
--output_dir "output/..."\
--amp 1 
```

---

## Citation

If you find our work to be useful in your research, please consider citing our paper:


```bibtex
@inproceedings{grethen2025StereoLunar,
  title={Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset},
  author={Grethen, Clémentine and Morin, Géraldine and Gasparini, Simone and Lebreton, Jérémy and Marti, Lucas and Gestido, Manuel Sanchez},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2025},
  month={October},
  note={Accepted at 3D-VAST Workshop}
}
```

---

##  Acknowledgements

This codebase builds upon many excellent open-source projects, such as MegaDepth, DUSt3R, CroCo, etc. We thank the respective authors for making their work publicly available.

This work was supported by the **European Space Agency (ESA)** under contract **4000140461/23/NL/GLC/my**.

We gratefully acknowledge the valuable support and collaboration of the **European Space Agency (ESA)** and **Airbus Defence and Space** for this work.
