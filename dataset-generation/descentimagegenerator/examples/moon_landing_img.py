#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from descentimagegenerator.config import (
    DIGConfig,
    SceneConfig,
    ServerConfig,
    SimulationConfig,
    OutputConfig,
    TrajectoryConfig,
)
from descentimagegenerator.trajectoryrenderer import TrajectoryRenderer, ImageFormat
from descentimagegenerator.display import DisplayVideo

if __name__ == "__main__":
    tesoa_traj = "/imagechain/data/space_cv/2021_EL3/livraisons/from_TESOA/20221212_northhemisphere_altitudeOK/2022-12-12_submarineIpLandingSite.mat"

    renderer = TrajectoryRenderer(
        config=DIGConfig(
            scene_config=SceneConfig(
                DEM_path="change2_20m.dem", body_brdf="hapke.brdf",
            ),
            trajectory_config=TrajectoryConfig(
                trajectory_path=tesoa_traj,
                sun_trajectory_path=tesoa_traj,
                sun_trajectory_config={"sun_index": 330},
            ),
            server_config=ServerConfig(
                hostname="shackleton",
                port=5113,
                resource_path="/mnt/ssd0/nmenga/surrender_data",
            ),
            simulation_config=SimulationConfig(),
            output_config=OutputConfig(),
        )
    )

    display = DisplayVideo("image")

    renderer.render_frame(1000)
    display(renderer.get_visible_frame(ImageFormat.Gray32F), renderer.get_depth_map())
    plt.show()
