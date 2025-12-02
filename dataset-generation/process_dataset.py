#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import cv2
import imageio.v3 as iio
from tqdm import tqdm

VALID_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif"}

def normalize_depth(depth, los_map):
    """Renvoie la carte Z corrigée (zeros hors plage ou LOS négatif)."""
    Z = depth * np.abs(los_map[..., 2])
    print(Z)
    return Z

def read_meta(npz_path: Path):
    data = np.load(npz_path)
    required = ["dmap", "los_map", "K", "R_w2c", "t_w2c"]
    if not all(k in data for k in required):
        missing = [k for k in required if k not in data]
        raise KeyError(f"{npz_path.name}: champs manquants {missing}")
    dmap    = data["dmap"].astype(np.float32)
    los_map = data["los_map"].astype(np.float32)
    K       = data["K"]
    R_w2c   = data["R_w2c"]
    t_w2c   = data["t_w2c"].reshape(3, 1)
    return dmap, los_map, K, R_w2c, t_w2c

def process_one(img_path: Path, npz_path: Path, out_dir: Path):
    tag = img_path.stem
    out_img   = out_dir / f"{tag}.jpg"
    out_depth = out_dir / f"{tag}.exr"
    out_meta  = out_dir / f"{tag}.npz"

    # lecture image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError("impossible de lire l’image")

    dmap, los_map, K, R_w2c, t_w2c = read_meta(npz_path)   

    Z = normalize_depth(dmap, los_map)

    # cam2world
    T_w2c = np.eye(4, dtype=np.float32)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3]  = t_w2c[:, 0]
    T_c2w = np.linalg.inv(T_w2c)

    R_in2out = np.eye(3, dtype=np.float32)
    T_c2w[:3, :3] = T_c2w[:3, :3] @ R_in2out.T

    cv2.imwrite(str(out_img), img_bgr)                 # JPG
    iio.imwrite(out_depth, Z.astype(np.float32), extension=".exr")
    np.savez(out_meta, intrinsics=K, cam2world=T_c2w)

def main(images_dir, metadata_dir, output_dir):
    img_dir  = Path(images_dir)
    meta_dir = Path(metadata_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in VALID_IMG_EXT)
    if not images:
        sys.exit("  Aucune image trouvée.")

    kept, skipped = 0, 0
    for img_path in tqdm(images):
        if img_path.suffix.lower() not in VALID_IMG_EXT:
            continue
        npz_path = meta_dir / img_path.with_suffix(".npz").name   # ← NEW
        print(npz_path)
        if not npz_path.exists():        
            skipped += 1
            continue
        try:
            process_one(img_path, npz_path, out_dir)
            kept += 1
        except Exception as e:
            print(f"  {img_path.stem} ignoré ({e})")
            skipped += 1


    print(f"\n  Terminé. {kept} paires gardées, {skipped} ignorées.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",   required=True, help="dossier images (.png, .jpg …)")
    ap.add_argument("--metadata_dir", required=True, help="dossier metadata (sous-dossiers par image)")
    ap.add_argument("--output_dir",   required=True, help="dossier de sortie nettoyé")
    args = ap.parse_args()

    main(args.images_dir, args.metadata_dir, args.output_dir)
