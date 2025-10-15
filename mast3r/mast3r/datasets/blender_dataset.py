"""
Dataset MASt3R <-> Blender
--------------------------
Lecture d’un dossier ``exports/`` généré par generate_dataset.py
et renvoi d’items au format attendu par dust3r_visloc / MASt3R.

▪ RGB          : cam0000_rgb.png
▪ Depth (opt.) : cam0000_depth0001.exr  (float32, distance métrique)
▪ Métadonnées  : cam0000.json           (K, R, t  –> repère CV)

Chaque __getitem__ renvoie une *paire* (query, map) :
    views = [query_view, map_view]
avec   query_view  sans pts3d,
       map_view    avec pts3d + mask « valid » (pour pixel_tol > 0).

Auteur : ChatGPT (avril 2025)
"""
import json, os, cv2, math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import torch ; a = torch.ones(1, device="cuda")
def load_depth_exr(path: Path) -> np.ndarray:
    """Lit un exr 32 bits mono-canal → np.float32 (H, W)."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Impossible de lire la depth : {path}")
    if depth.ndim == 3:  # cv2 lit parfois (H,W,3)  – on garde la 1ʳᵉ couche
        depth = depth[..., 0]
    return depth.astype(np.float32)


def world_to_cam_to_cam_to_world(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """
    Inverse [R | t] (world->cam) -> 4×4 cam->world.
    """
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc.reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_cw
    T[:3, 3] = t_cw
    return T


def depth_to_pts3d(depth: np.ndarray, K: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Projette une depth métrique en points 3D (repère caméra CV).
    Renvoie pts3d (H,W,3) et masque booléen valid.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # grille de pixels
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)            # (H,W)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack((X, Y, Z), axis=-1)    # (H,W,3)

    valid = Z > 0                        # masque profondeur valide
    return pts, valid


class BlenderMASt3RDataset(Dataset):
    """
    Chaque élément = une paire (query, map) issue d’un même couple A/B.
    """
    def __init__(self, root_dir: str, load_depth: bool = True):
        self.root = Path(root_dir)
        assert self.root.is_dir(), f"{root_dir} n’existe pas"
        self.load_depth = load_depth

        # On récupère les couples camXXXX_rgb / json
        rgb_paths = sorted(self.root.glob("cam*_rgb.png"))
        assert len(rgb_paths) % 2 == 0, "Il doit y avoir un nombre pair d’images"

        # On groupe les images 2 par 2 (A puis B dans ton generate_dataset.py)
        self.pairs = []
        for i in range(0, len(rgb_paths), 2):
            self.pairs.append([rgb_paths[i], rgb_paths[i + 1]])

    # --- utilitaire interne -------------------------------------------------
    def _build_view(self, rgb_path: Path, with_pts: bool):
        stem = rgb_path.stem.replace("_rgb", "")
        meta_path = self.root / f"{stem}.json"
        depth_path = self.root / f"{stem}_depth0001.exr"

        # -- RGB
        img = Image.open(rgb_path).convert("RGB")

        # -- Métadonnées
        with open(meta_path) as f:
            meta = json.load(f)
        K = np.asarray(meta["K"], dtype=np.float32)
        R_wc = np.asarray(meta["R"], dtype=np.float32)
        t_wc = np.asarray(meta["t"], dtype=np.float32)

        cam_to_world = world_to_cam_to_cam_to_world(R_wc, t_wc)

        view = dict(
            image_name=rgb_path.name,
            rgb=img,
            intrinsics=K,
            cam_to_world=cam_to_world,
            distortion=None,   # pas de distorsion
        )

        if with_pts and self.load_depth and depth_path.exists():
            depth = load_depth_exr(depth_path)
            pts3d, valid = depth_to_pts3d(depth, K)
            view["depth"] = depth
            view["pts3d"] = torch.from_numpy(pts3d)      # (H,W,3)
            view["valid"] = torch.from_numpy(valid)      # (H,W)  bool
        # ------------------------------------------------------------------
        # 1) image mise à l’échelle du modèle  (même résolution que l’original pour l’instant)
        rgb_np = np.array(img)                              # (H,W,3) uint8
        rgb_tensor = torch.from_numpy(rgb_np).permute(2,0,1).float() / 255.0
        rgb_tensor = 2.0 * rgb_tensor - 1.0                 # dans [-1,1]
        view["rgb_rescaled"] = rgb_tensor                   # (3,H,W)  torch.float32

        # 2) si on a déjà un masque + pts3d, on copie pour la version _rescaled
        if "valid" in view:
            view["valid_rescaled"] = view["valid"]          # même H,W
        if "pts3d" in view:
            view["pts3d_rescaled"] = view["pts3d"]          # (H,W,3)
        # ------------------------------------------------------------------
# ------------------------------------------------------------------
        # Transformation (colmap→orig) : ici identité car on ne redimensionne pas
        view["to_orig"] = np.eye(3, dtype=np.float32)   # (3×3)
        # ------------------------------------------------------------------

        return view

    # -----------------------------------------------------------------------
    def __getitem__(self, idx):
        rgb_q, rgb_m = self.pairs[idx]

        # premier = query (pas besoin de pts3d) ; second = map (avec)
        query_view = self._build_view(rgb_q, with_pts=False)
        map_view   = self._build_view(rgb_m, with_pts=True)

        return [query_view, map_view]     # format attendu par visloc

    def __len__(self):
        return len(self.pairs)

    # MASt3R appelle parfois cette méthode
    def set_resolution(self, model):
        pass    # rien à faire (on laisse MASt3R choisir la rés. dynamique)
