import os.path as osp
import numpy as np
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class LunarDataset(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, split_file, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.split_file = split_file
        self._load_data(split_file)

        print(f"[LunarDataset] {len(self.pairs)} pairs from {len(self.all_scenes)} scene(s)")

    def _load_data(self, split_file):
        data = np.load(osp.join(self.ROOT, split_file), allow_pickle=True)
        self.all_scenes = data['scenes']
        self.all_images = data['images']
        self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]
        scene     = self.all_scenes[scene_id]            # « moon_scene_0000 »
        seq_path  = osp.join(self.ROOT, scene)

        views = []
        for im_id in [im1_id, im2_id]:
            img_name = self.all_images[im_id]
            try:
                img   = imread_cv2(osp.join(seq_path, img_name + ".jpg"))
                depth = imread_cv2(osp.join(seq_path, img_name + ".exr")).astype(np.float32)
                params = np.load(osp.join(seq_path, img_name + ".npz"))
            except Exception as e:
                # → fera « skip » : le DataLoader passe au sample suivant
                raise IndexError(f"{scene}/{img_name} illisible : {e}")

            # --- poses / intrinsics --------------------------------------
            K = params["intrinsics"].astype(np.float32)
            T = params["cam2world"].astype(np.float32)

            # --- nettoyage & clipping -----------------------------------
            depth[~np.isfinite(depth)] = 0.0           # remplace les NaN/Inf
            depth[depth <= 0]          = 0.0

            valid = depth[depth > 0]
            if valid.size:                               # clip sur valeurs valides
                clip_val = np.percentile(valid, 98)
                depth[depth > clip_val] = 0.0
            else:
                # depth totalement vide → on ignore la paire
                raise IndexError(f"{scene}/{img_name} depthmap vide")

            # --- resize / crop ------------------------------------------
            img, depth, K = self._crop_resize_if_necessary(
                img, depth, K, resolution, rng, info=(scene, img_name)
            )

            # --- sky mask = pixels invalides après resize ---------------
            sky_mask = (~np.isfinite(depth)) | (depth <= 0)

            # --- assemble la vue ----------------------------------------
            views.append(dict(
                img=img,
                depthmap=depth,
                camera_pose=T,
                camera_intrinsics=K,
                dataset="Lunar",
                label=scene,
                instance=img_name,
                is_metric_scale=True,
                sky_mask=sky_mask
            ))

        return views