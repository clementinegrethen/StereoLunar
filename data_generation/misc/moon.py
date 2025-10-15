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
        scene = self.all_scenes[scene_id]  # always "moon_scene_0000"
        seq_path = osp.join(self.ROOT, scene)

        views = []
        for im_id in [im1_id, im2_id]:
            img_name = self.all_images[im_id]

            try:
                img = imread_cv2(osp.join(seq_path, img_name + '.jpg'))
                depth = imread_cv2(osp.join(seq_path, img_name + '.exr')).astype(np.float32)
                params = np.load(osp.join(seq_path, img_name + '.npz'))
            except Exception as e:
                raise RuntimeError(f" Impossible to load : {img_name} : {e}")

            K = params["intrinsics"].astype(np.float32)
            T = params["cam2world"].astype(np.float32)

            # Nettoyage optionnel
            depth[depth > np.percentile(depth[depth > 0], 98)] = 0.0

            img, depth, K = self._crop_resize_if_necessary(
                img, depth, K, resolution, rng, info=(scene, img_name)
            )

            views.append(dict(
                img=img,
                depthmap=depth,
                camera_pose=T,
                camera_intrinsics=K,
                dataset="Lunar",
                label=scene,
                instance=img_name
            ))

        return views
