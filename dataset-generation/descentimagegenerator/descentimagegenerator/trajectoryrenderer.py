from logging import Logger
from enum import Enum, auto
from typing import Callable, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from descentimagegenerator.config import DIGConfig, ua
from descentimagegenerator.trajectoryreader import (
    TrajectoryReader,
    DefaultTrajectoryReader,
    Trajectory,
)
from descentimagegenerator.logger import get_logger
from surrender.surrender_client import surrender_client


class ImageFormat(Enum):
    Gray8 = auto()
    Gray32F = auto()


class TrajectoryRenderer:
    def __init__(
            self,
            config: DIGConfig,
            trajectory_reader: TrajectoryReader = DefaultTrajectoryReader(),
            cam_traj: Optional[Trajectory] = None,
            sun_traj: Optional[Trajectory] = None,
            logger: Logger = get_logger("TrajectoryRenderer"),
    ):
        self.boulder_sizes = None
        self.use_sensor = None
        self.boulder_meshes = None
        self.boulder_positions = None
        self.boulder_attitudes = None
        self.logger = logger

        self.config = config

        self.debug = False
        self.positions: np.ndarray
        self.rotation: np.ndarray

        self.client: surrender_client = surrender_client()
        self.client.connectToServer(
            config.server_config.hostname, config.server_config.port
        )

        self.client.setTimeOut(config.server_config.timeout)
        if type(config.server_config.resource_path) is str:
            resourcepath = [config.server_config.resource_path]
        else:
            resourcepath = config.server_config.resource_path
        for path in resourcepath:
            self.client.cd(path)
           #self.client.pushResourcePath()
        self.client.setRessourcePath(config.server_config.resource_path)
        self.client.setConventions(
            self.client.SCALAR_XYZ_CONVENTION, self.client.Z_FRONTWARD
        )
        self.client.enableRegularPSFSampling(True)
        self.client.enablePhotonAccumulation(False)
        self.client.setSelfVisibilitySamplingStep(5)
        self.client.setPhotonMapSamplingStep(5)
        self.client.setMaxSecondaryRays(1)
        self.client.enableRaytracing(True)
        self.client.enablePathTracing(False)
        self.client.enablePreviewMode(False)
        self.client.setNbSamplesPerPixel(config.simulation_config.rays_per_pixels)
        self.client.enableLabelMapping(True)
        self.client.enableVarianceMapping(True)
        self.client.setStdDevThreshold(4e-13)
        self.image_shape = config.simulation_config.image_shape
        self.client.setImageSize(*self.image_shape)
        self.client.enableIrradianceMode(False)

        # Set the scene
        # Load lua file defining textures for hybrid DEM if exists
        if config.scene_config.hybrid_textures_lua_path:
            self.client.runLuaScript(config.scene_config.hybrid_textures_lua_path)

        self.setup_body()
        self.setup_sun()

        if self.config.boulder_config.coordinates_file_path != "":
            self.add_boulders()

        # open camera trajectory
        if config.trajectory_config.trajectory_path is not None:
            self.cam_traj = trajectory_reader.read_trajectory(
                config.trajectory_config.trajectory_path,
                config.trajectory_config.trajectory_config,
            )
        else:
            assert cam_traj is not None
            self.cam_traj = cam_traj

        # open sun trajectory
        if config.trajectory_config.sun_trajectory_path is not None:
            self.sun_traj = trajectory_reader.read_trajectory(
                config.trajectory_config.sun_trajectory_path,
                config.trajectory_config.sun_trajectory_config,
                is_sun=True,
            )
        else:
            assert sun_traj is not None
            self.sun_traj = sun_traj

        self.setup_sensor()

    def setup_sensor(self):
        """
        Defining the sensor. Use setup_sensor.lua file in priority instead of FOV and PSF
        :return:
        """

        if self.config.simulation_config.sensor_lua != "":
            if not Path(self.config.simulation_config.sensor_lua).exists():
                self.logger.error("Could not find sensor lua file")
                exit(1)
            with open(self.config.simulation_config.sensor_lua, "r") as sensor_file:
                self.logger.info(
                    "Using sensor lua file. Config integration time and FoV aren't used"
                )
                sensor_file_name = Path(self.config.simulation_config.sensor_lua).name
                sensor_file_content = sensor_file.read()
                self.client.sendFile(sensor_file_name, sensor_file_content)
                self.client.runLuaScript(sensor_file_name)
                self.use_sensor = True
        else:
            self.use_sensor = False
            self.client.setCameraFOVDeg(*self.config.simulation_config.cam_fov_deg)
            self.client.setIntegrationTime(
                self.config.simulation_config.cam_integration_time
            )

            if self.config.simulation_config.psf_name is not None:
                self.client.loadPSFModel(
                    f"{self.config.simulation_config.psf_name}.psf",
                    self.config.simulation_config.psf_args,
                )

    def setup_body(self):
        """
        Set body parameters
        :return:
        """

        self.logger.info("Setting up the body")

        self.client.createBRDF(
            "body_brdf",
            self.config.scene_config.body_brdf,
            self.config.scene_config.body_brdf_args,
        )
        _dem_infos = self.client.createSphericalDEM(
            "body",
            self.config.scene_config.DEM_path,
            "body_brdf",
            self.config.scene_config.albedo_path,
        )

    def setup_sun(self):
        """
        setup sun parameters
        :return:
        """

        self.logger.info("Setting up the Sun")

        self.client.createBRDF("sun", "sun.brdf", {})
        self.client.createShape("sun_shape", "sphere.shp", {"radius": 696342000})
        self.client.createBody("sun", "sun_shape", "sun", {})
        self.client.setSunPower([self.config.scene_config.sun_power] * 4)
        self.client.setObjectPosition("sun", [50*ua, 0, 0]) #Put the sun very far away to avoid issues when projecting meshes

    def add_boulders(self):
        """
        Add boulders to the scene
        :return:
        """

        self.logger.info("Adding boulders")

        coordinates_df = pd.read_parquet(self.config.boulder_config.coordinates_file_path)
        self.logger.debug("Parquet read, %d records", len(coordinates_df.index))

        self.boulder_positions = coordinates_df[["position_x_bcbf", "position_y_bcbf", "position_z_bcbf"]].to_numpy()
        self.boulder_attitudes = coordinates_df[["attitude_q0", "attitude_qx", "attitude_qy", "attitude_qz"]].to_numpy()
        self.boulder_meshes = coordinates_df["filename"].to_list()
        self.boulder_sizes = coordinates_df["size"].to_numpy()

        self.boulder_attitudes = self.config.boulder_config.scale_factor * \
                                 self.boulder_sizes[:, np.newaxis] * self.boulder_attitudes

        if self.config.boulder_config.project_boulders:
            self.logger.debug("Projecting boulders")

            self.client.loadAndProjectMultipleMeshes(
                "boulders",
                self.boulder_meshes,
                self.boulder_positions,
                self.boulder_attitudes,
                self.config.boulder_config.projection_center,
                self.config.boulder_config.offset_ratio,
            )
        else:
            self.client.loadMultipleMeshes(
                "boulders",
                self.boulder_meshes,
                self.boulder_positions,
                self.config.boulder_config.attitudes,
            )

        self.logger.debug("Projection done !")

        self.client.createBRDF(
            "boulder_brdf",
            self.config.boulder_config.brdf,
            self.config.boulder_config.brdf_args,
        )
        self.client.setObjectElementBRDF("boulders", "boulders", "boulder_brdf")
        if self.config.boulder_config.attach_objects:
            self.client.attachObjects("body", ["boulders"])

        self.client.setObjectLabel("boulders", self.config.boulder_config.labels)

    def get_trajectory_length(self) -> Optional[int]:
        return self.cam_traj.get_length()

    def update_objects_positions(self, frame_index: int):
        self.client.setObjectPosition("sun", self.sun_traj.get_position(frame_index))
        self.client.setObjectPosition("camera", self.cam_traj.get_position(frame_index))
        self.client.setObjectAttitude("camera", self.cam_traj.get_rotation(frame_index))

    def render_frame(
            self,
            frame_index: int,
            callback: Optional[Callable[[surrender_client, int], None]] = None,
    ):
        self.update_objects_positions(frame_index)

        if self.use_sensor:
            self.client.runLuaCode("sensor:render();")
        else:
            self.client.render()

        if callback is not None:
            callback(self.client, frame_index)

        return True

    def get_visible_frame(
            self, img_format: ImageFormat = ImageFormat.Gray8
    ) -> np.ndarray:
        return getattr(self.client, f"getImage{img_format.name}")()

    def get_depth_map(self) -> np.ndarray:
        return self.client.getDepthMap()

    def save_visible_frame(
            self, filename: str, img_format: ImageFormat = ImageFormat.Gray8
    ) -> None:
        return getattr(self.client, f"saveImage{img_format.name}")(filename)
