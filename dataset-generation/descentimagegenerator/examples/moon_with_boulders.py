#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from descentimagegenerator.config import (
    DIGConfig,
    SceneConfig,
    ServerConfig,
    SimulationConfig,
    OutputConfig,
    BoulderConfig,
    TrajectoryConfig,
)
from descentimagegenerator.trajectoryrenderer import TrajectoryRenderer, ImageFormat
from descentimagegenerator.logger import get_logger

from surrender.geometry import look_at, normalize

MOON_RADIUS = 1_737_400  # in meter
UA = 1.49597870700e11  # (m)


def render_images(renderer: TrajectoryRenderer, image_step: int, logger):
    output_dir = Path(renderer.config.output_config.output_directory)
    (output_dir / "images").mkdir(exist_ok=True)
    idx_range = np.arange(0, renderer.cam_traj.get_length(), image_step)
    for img_idx in tqdm(idx_range):
        logger.debug(f"rendering {img_idx=}")

        renderer.client.setObjectPosition(
            "sun", renderer.sun_traj.get_position(img_idx)
        )
        cam_pos = renderer.cam_traj.get_position(img_idx)
        renderer.client.setObjectPosition(
            "camera", cam_pos
        )  ## initial position at R_MOON +10000 (m)
        look_at(
            renderer.client, cam_pos, [0.0, 0.0, 0.0]
        )  ## point to the center of the moon

        renderer.client.render()
        img = renderer.get_visible_frame(img_format=ImageFormat.Gray32F)
        logger.debug(f"{img.min()=} {img.max()=}")
        # normalize image for 8 bit PNG
        img8b = (
            255 * np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)
        ).astype(np.uint8)
        Image.fromarray(img8b).save(output_dir / "images" / f"{img_idx:05d}.png")
        # cv2.imwrite(str(output_dir / 'images'/ f'{img_idx:05d}.tiff'), img)


def render_metadata(renderer_meta: TrajectoryRenderer, metadata_step: int, logger):

    output_dir = Path(renderer_meta.config.output_config.output_directory)
    for dir in ["images", "dmap", "slopes", "labels", "intersect"]:
        (output_dir / dir).mkdir(exist_ok=True)

    with open(
        output_dir / "intersect" / f"simulation_intersect.csv", "w"
    ) as f_intersect:
        idx_range = np.arange(0, renderer_meta.cam_traj.get_length(), metadata_step)
        for img_idx in tqdm(idx_range):
            logger.debug(f"computing label map, depthmap and slopes for {img_idx=}")
            renderer_meta.client.setObjectPosition(
                "sun", renderer_meta.sun_traj.get_position(img_idx)
            )
            cam_pos = renderer_meta.cam_traj.get_position(img_idx)
            renderer_meta.client.setObjectPosition("camera", cam_pos)
            look_at(renderer_meta.client, cam_pos, [0.0, 0.0, 0.0])
            renderer_meta.client.render()
            label_map = renderer_meta.client.getLabelMap()
            dmap = renderer_meta.client.getDepthMap()
            nmap = renderer_meta.client.getNormalMap()
            assert label_map is not None and nmap is not None and dmap is not None

            print(
                f"{img_idx:05d}, {renderer_meta.client.intersectScene([(cam_pos, -normalize(cam_pos))])[0]}",
                file=f_intersect,
            )
            Image.fromarray(dmap).save(output_dir / "dmap" / f"{img_idx:05d}.tiff")

            normal_vecs = np.vstack(
                (
                    nmap[:, :, 0].reshape(1, -1),
                    nmap[:, :, 1].reshape(1, -1),
                    nmap[:, :, 2].reshape(1, -1),
                )
            )
            angle_normal = np.arcsin(
                np.linalg.norm(
                    np.cross(
                        normal_vecs.transpose(),
                        np.tile(np.array([0, 0, -1]), (1024 * 1024, 1)),
                    ),
                    axis=1,
                )
            ).reshape(1024, 1024)
            Image.fromarray(np.rad2deg(angle_normal).astype(float)).save(
                output_dir / "slopes" / f"{img_idx:05d}.tiff"
            )
            Image.fromarray(label_map.astype(np.uint8)).save(
                output_dir / "labels" / f"{img_idx:05d}.png"
            )


def run_simulations(
    surrender_host: str,
    surrender_port: int,
    surrender_timeout: float,
    metadata_step: int,
    skip_metadata: bool,
    image_step: int,
    image_rays: int,
    skip_image: bool,
    add_boulders: bool,
):

    logger = get_logger("simulation.py")

    # connect to SurRender
    scene_config = SceneConfig(
        DEM_path="LOLA_southpole_5m.dem",
        body_brdf="hapke.brdf",
        body_brdf_args={"albedo": [0.12, 0.12, 0.12, 0.12]},
    )
    trajectory_config = TrajectoryConfig(
        trajectory_path="template_trajectory.csv",
        sun_trajectory_path="template_trajectory.csv",
    )
    server_config = ServerConfig(
        hostname=surrender_host,
        port=surrender_port,
        resource_path=f"{os.path.dirname(os.path.abspath(__file__))}",
        timeout=surrender_timeout,
    )

    boulder_config = BoulderConfig()
    if add_boulders:
        path_rock_db = Path(f"{os.path.dirname(os.path.abspath(__file__))}")
        conf_boulders = pd.read_csv(path_rock_db / "conf_boulders.csv")

        n_boulders = conf_boulders.shape[0]
        sizes_ = conf_boulders["sizes"].tolist()
        meshes_ = ["boulder.obj" for i in range(0, n_boulders)]
        attitudes_ = [
            (
                conf_boulders["q0"][i],
                conf_boulders["qx"][i],
                conf_boulders["qy"][i],
                conf_boulders["qz"][i],
            )
            for i in range(0, n_boulders)
        ]
        positions_ = [
            (conf_boulders["x"][i], conf_boulders["y"][i], conf_boulders["z"][i])
            for i in range(0, n_boulders)
        ]

        boulder_config = BoulderConfig(
            resource_path=f"{os.path.dirname(os.path.abspath(__file__))}",
            project_boulders=True,
            attach_objects=True,
            scale_factor=100.0,
            brdf="hapke.brdf",
            brdf_args={"albedo": [0.12, 0.12, 0.12, 0.12]},
            positions=positions_,
            attitudes=attitudes_,
            meshes=meshes_,
            sizes=sizes_,
            projection_center=(0.0, 0.0, 0.0, 1.0),
            offset_ratio=0.0,
        )

    logger.debug(f"running scenario")
    if not skip_metadata:
        renderer_meta = TrajectoryRenderer(
            config=DIGConfig(
                scene_config=scene_config,
                trajectory_config=trajectory_config,
                server_config=server_config,
                output_config=OutputConfig(),
                boulder_config=boulder_config,
                simulation_config=SimulationConfig(
                    rays_per_pixels=1,  # must remain 1 when generating metadata
                    image_shape=(1024, 1024),
                    cam_integration_time=1,
                    cam_fov_deg=(70.0, 70.0),
                ),
            )
        )
        renderer_meta.client.enableLabelMapping(True)
        renderer_meta.client.setObjectLabel("body", 1)
        if add_boulders:
            renderer_meta.client.setObjectLabel("boulder", 2)
        render_metadata(renderer_meta, metadata_step, logger)

    if not skip_image:
        renderer = TrajectoryRenderer(
            config=DIGConfig(
                scene_config=scene_config,
                trajectory_config=trajectory_config,
                server_config=server_config,
                output_config=OutputConfig(),  # output_directory=output_dir
                boulder_config=boulder_config,
                simulation_config=SimulationConfig(
                    rays_per_pixels=image_rays,
                    image_shape=(1024, 1024),
                    cam_fov_deg=(70.0, 70.0),
                    cam_integration_time=1,
                    psf_name="gaussian",
                    psf_args=dict(sigma=0.6),
                ),
            )
        )
        render_images(renderer, image_step, logger)


if __name__ == "__main__":

    run_simulations(
        surrender_host="localhost",
        surrender_port=5151,
        surrender_timeout=120.0,
        metadata_step=1,
        skip_metadata=True,
        image_step=1,
        image_rays=2,
        skip_image=False,
        add_boulders=True,
    )
