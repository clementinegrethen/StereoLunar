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
    CE3_traj = "/imagechain/data/space_cv/2021_EL3/from_nicolas/traj_reelles/ce3.csv"

    renderer = TrajectoryRenderer(
        config=DIGConfig(
            scene_config=SceneConfig(
                DEM_path="change2_20m_fused.dem",
                body_brdf="hapke.brdf",
                hybrid_textures_lua_path="conf/CE3_demonly.lua",
            ),
            trajectory_config=TrajectoryConfig(
                trajectory_path=CE3_traj,
                sun_trajectory_path=CE3_traj,
                sun_trajectory_config={"sun_index": 330},
            ),
            server_config=ServerConfig(
                hostname="shackleton",
                port=5113,
                resource_path="/mnt/ssd0/nmenga/surrender_data",
                timeout=600.0,
            ),
            simulation_config=SimulationConfig(),
            output_config=OutputConfig(),
        )
    )

    display = DisplayVideo("image")

    renderer.render_frame(400)
    display(renderer.get_visible_frame(ImageFormat.Gray32F), renderer.get_depth_map())
    plt.show()
