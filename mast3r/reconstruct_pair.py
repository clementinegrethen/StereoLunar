#!/usr/bin/env python3
"""StereoLunar — direct stereo reconstruction from one image pair.

This is the recommended way to evaluate MOONSt3R on your own image pairs.
It outputs the network's RAW two-view prediction: both views are predicted
directly in the first camera's frame, so NO pose optimization is involved —
what you see is exactly what the model produces.

Why prefer this over the interactive Gradio demo:
  * the Gradio demo falls back to the generic MASt3R weights unless you pass
    `--weights checkpoints/StereoLunar.pth` — the generic weights fail on
    lunar imagery (descriptor matching breaks on the low-contrast texture,
    and the demo's pose optimization then diverges, producing two separated
    point-cloud sheets);
  * the demo re-estimates camera poses from descriptor matches on top of the
    network output, which adds hyperparameters (matching confidence, learning
    rates) that need tuning on high-altitude pairs.

Usage (from the mast3r/ directory):
    python reconstruct_pair.py \
        ../quick_testing/Nadir_baseline_sample/paire1/im_00594.jpg \
        ../quick_testing/Nadir_baseline_sample/paire1/im_00595.jpg \
        --weights checkpoints/StereoLunar.pth

Outputs: out/fused.ply (colored point cloud, both views in camera-1 frame),
out/preview.png, and a cross-view consistency score.

Notes:
  * Weights: https://huggingface.co/cgrethen/MOONSt3R (auto-downloaded if
    `--weights` is omitted and no local checkpoint is found).
  * The output is in the network's own scale. For very high-altitude pairs
    (e.g. the Nadir samples, ~30 km) the absolute scale is not observable
    from two views: the reconstruction is internally consistent, but its
    global size is arbitrary. The quantitative evaluation in the paper
    aligns the fused scene to the ground truth with a single similarity
    transform before computing metrics.
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, 'dust3r')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference

DEFAULT_LOCAL = os.path.join(_ROOT, 'checkpoints', 'StereoLunar.pth')
DEFAULT_HF = 'cgrethen/MOONSt3R'


def load_model(weights, device):
    """Load either a released model file, a training checkpoint
    ({'args': ..., 'model': ...}), or a Hugging Face model id."""
    if os.path.isfile(weights):
        ckpt = torch.load(weights, map_location='cpu')
        if isinstance(ckpt, dict) and 'args' in ckpt and 'model' in ckpt:
            spec = ckpt['args'].model.replace('ManyAR_PatchEmbed', 'PatchEmbedDust3R')
            if 'landscape_only' not in spec:
                spec = spec[:-1] + ', landscape_only=False)'
            else:
                spec = spec.replace(' ', '').replace('landscape_only=True',
                                                     'landscape_only=False')
            inf = float('inf')  # noqa: F841 — used inside the model spec string
            model = eval(spec)  # noqa: S307 — spec comes from the checkpoint itself
            model.load_state_dict(ckpt['model'], strict=False)
            return model.to(device).eval()
        del ckpt
    return AsymmetricMASt3R.from_pretrained(weights).to(device).eval()


def write_ply(path, pts, colors):
    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n'
                f'element vertex {len(pts)}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                'end_header\n')
        for p, c in zip(pts, colors):
            f.write(f'{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image1')
    ap.add_argument('image2')
    ap.add_argument('--weights', default=None,
                    help='checkpoint (.pth) or HF id; default: '
                         'checkpoints/StereoLunar.pth, else auto-download '
                         f'{DEFAULT_HF}')
    ap.add_argument('--out', default='out', help='output directory')
    ap.add_argument('--size', type=int, default=512, help='inference resolution')
    ap.add_argument('--device', default=None, help='cuda / cpu (default: auto)')
    ap.add_argument('--min-conf', type=float, default=0.0,
                    help='drop points with confidence below this value (0 = keep all)')
    args = ap.parse_args()

    weights = args.weights or (DEFAULT_LOCAL if os.path.isfile(DEFAULT_LOCAL)
                               else DEFAULT_HF)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    print(f'Loading model from {weights} ({device}) ...')
    model = load_model(weights, device)

    imgs = load_images([args.image1, args.image2], size=args.size, verbose=False)
    pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
    with torch.no_grad():
        out = inference(pairs, model, device, batch_size=1, verbose=True)

    # Direction where view1 == image1. Both predictions live in camera-1 frame.
    idx = 0 if out['view1']['idx'][0] == 0 else 1
    p1 = out['pred1']['pts3d'][idx].cpu().numpy()                # (H, W, 3)
    p2 = out['pred2']['pts3d_in_other_view'][idx].cpu().numpy()  # (H, W, 3)
    c1 = out['pred1']['conf'][idx].cpu().numpy()
    c2 = out['pred2']['conf'][idx].cpu().numpy()
    im1 = (out['view1']['img'][idx].permute(1, 2, 0).cpu().numpy() + 1) / 2
    im2 = (out['view2']['img'][idx].permute(1, 2, 0).cpu().numpy() + 1) / 2

    m1 = c1 >= args.min_conf
    m2 = c2 >= args.min_conf
    pts = np.vstack([p1[m1], p2[m2]])
    cols = np.vstack([im1[m1], im2[m2]])

    ply = os.path.join(args.out, 'fused.ply')
    write_ply(ply, pts, cols)
    print(f'Fused cloud ({len(pts)} pts, both views in camera-1 frame) -> {ply}')

    # Cross-view consistency: NN distance between the two sheets, % of depth.
    try:
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(0)
        A = p1.reshape(-1, 3)[rng.choice(p1.size // 3, 20000, replace=False)]
        B = p2.reshape(-1, 3)[rng.choice(p2.size // 3, 20000, replace=False)]
        dnn, _ = cKDTree(B).query(A, k=1)
        mis = np.median(dnn) / max(np.median(p1[..., 2]), 1e-9) * 100
        verdict = ('OK' if mis < 1.0 else
                   'poor — make sure you loaded the StereoLunar checkpoint, '
                   'not the generic MASt3R weights')
        print(f'Cross-view consistency: median inter-sheet distance = '
              f'{mis:.2f}% of scene depth ({verdict})')
    except ImportError:
        pass

    # Preview figure (inputs, depth, fused cloud).
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 7))
        for i, (im, t) in enumerate([(im1, 'view 1'), (im2, 'view 2')]):
            ax = fig.add_subplot(2, 3, 1 + i)
            ax.imshow(im); ax.set_title(t); ax.axis('off')
        ax = fig.add_subplot(2, 3, 3)
        d = ax.imshow(p1[..., 2], cmap='turbo')
        ax.set_title('depth (view 1)'); ax.axis('off')
        plt.colorbar(d, ax=ax, fraction=0.04)
        sub = 6
        P1, P2 = p1[::sub, ::sub].reshape(-1, 3), p2[::sub, ::sub].reshape(-1, 3)
        C1, C2 = im1[::sub, ::sub].reshape(-1, 3), im2[::sub, ::sub].reshape(-1, 3)
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        ax.scatter(*P1.T, s=0.3, c='red', alpha=.5, label='view 1')
        ax.scatter(*P2.T, s=0.3, c='blue', alpha=.5, label='view 2')
        ax.legend(markerscale=20); ax.view_init(35, -60); ax.set_axis_off()
        ax.set_title('fused (colored by view)')
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        ax.scatter(*np.vstack([P1, P2]).T, s=0.3,
                   c=np.clip(np.vstack([C1, C2]), 0, 1))
        ax.view_init(35, -60); ax.set_axis_off(); ax.set_title('fused (textured)')
        fig.tight_layout()
        png = os.path.join(args.out, 'preview.png')
        fig.savefig(png, dpi=130)
        print(f'Preview -> {png}')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
