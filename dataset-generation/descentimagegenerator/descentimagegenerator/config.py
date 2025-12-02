from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Tuple, Optional, Dict, Union

import numpy as np

ua = 149_597_870_700


@dataclass_json
@dataclass
class ServerConfig:
    """ SurRender Server config

    Args:
        hostname (str): surrender server hostname or ip as string
        port (int): surrender server port, better avoid the default port (`5151`) unless you want to break someone else's simulation or your own...
        resource_path (str): surrender server resource path
        timeout (float): surrender server timeout duration in seconds, you may wish to increase this value if you expect long rendering times
    """

    hostname: str = "127.0.0.1"
    port: int = 2000
    resource_path: Union[str, List[str]] = ""
    timeout: float = 30.0  # s


@dataclass_json
@dataclass
class SceneConfig:
    """ Scene Config

    Args:
        DEM_path (str): path to ".dem" file, please note that it should be accessible by surrender server       
        body_brdf (str): name of the BRDF, e.g. "raw.brdf" or "hapke.brdf"
        body_brdf_args (Optional[Dict]): optional additional arguments as a dict for the brdf model
        albedo_path (str): path to albedo texture, e.g. "default.png"
        body_radius (float): body radius (m)
        sun_power (float): it is probably going to be wrong the first time you set it...
        hybrid_textures_lua_path (str): optional lua file that describes hybrid DEM, see examples
    """

    DEM_path: str = ""
    body_brdf: str = ""
    body_brdf_args: Optional[Dict] = field(default_factory=dict)
    albedo_path: str = ""
    body_radius: float = 1_737_400
    sun_power: float = 10 * ua * ua * np.pi * 5.2 * 5.2
    hybrid_textures_lua_path: str = ""


@dataclass_json
@dataclass
class TrajectoryConfig:
    """ Trajectory Config

    Args:
        trajectory_path (Optional[str]): path to trajectory file (several formats are supported, see ./trajectoryreader.py)
        sun_trajectory_path (Optional[str]): path to Sun trajectory file
        sun_trajectory_config (Optional[Dict]): optional additional data to provide to the sun trajectory reader
        trajectory_config (Optional[Dict]): optional additional data to provide to the trajectory reader
    """

    trajectory_path: Optional[str] = None
    sun_trajectory_path: Optional[str] = None
    sun_trajectory_config: Optional[Dict] = field(default_factory=dict)
    trajectory_config: Optional[Dict] = field(default_factory=lambda: {"type": "BCBF"})


@dataclass_json
@dataclass
class SimulationConfig:
    """ Simulation Config

    Args:
        rays_per_pixels (int): number of rays per pixel, higher values mean longer rendering time but less noise
        sensor_lua: (str): optional sensor model file path such as GenericSensor.lua
        image_shape (Tuple[int, int]): width, height (pixels) of the image to render
        camera_parameters_path (str): looks unused for now
        cam_fov_deg (Tuple[float, float]): camera fov in degrees (fov_x, fov_y)
        cam_integration_time (float): integration time in seconds (beware, it could behave differently when a sensor.lua is provided)
        cam_auto_integration (bool): enable automatic integration time
        cam_auto_integration_rate (float): how fast the integration time may be updated when using cam_auto_integration
        cam_auto_integration_bounds (Tuple[float, float]): bounds when enabling cam_auto_integration
        psf_name (str): e.g. "gaussian"
        psf_args (Dict): optional additional parameters for the psf model
        depth_map_size_factor (float): looks unused for now
    """

    rays_per_pixels: int = 4
    sensor_lua: str = ""
    image_shape: Tuple[int, int] = (1024, 1024)
    camera_parameters_path: str = ""
    cam_fov_deg: Tuple[float, float] = (60.0, 60.0)
    cam_integration_time: float = 1e-4
    cam_auto_integration: bool = False
    cam_auto_integration_rate: float = 0.1
    cam_auto_integration_bounds: Tuple[float, float] = (0.8, 0.95)
    psf_name: Optional[str] = None
    psf_args: Dict = field(default_factory=dict)
    depth_map_size_factor: float = 1.0


@dataclass_json
@dataclass
class OutputConfig:
    """ Output Config

    Args:
        output_directory (str): path to the output directory
        image_range (Tuple[int, int]): image id min, max
        step (int): step for trajectory
        map_filename (str): looks unused for now
        map_gsd (float): looks unused for now
        copy_trajectory_file (bool): copy trajectory file to output directory
    """

    output_directory: str = ""
    image_range: Tuple[int, int] = (0, -1)
    step: int = 1
    map_filename: str = ""
    map_gsd: float = 20.0
    copy_trajectory_file: bool = False


@dataclass_json
@dataclass
class BoulderConfig:
    """ Boulder Config

    Args:
        project_boulders (bool): use SurRender to load and project meshes on a projection center
        attach_objects (bool): useful if the object the boulders are projected on moves
        scale_factor (float): scaling to apply on projected objects
        brdf (str): objects brdf e.g. "hapke.brdf"
        brdf_args (Optional[Dict]): optional additional parameters for the brdf model
        coordinates_file_path (str): path to parquet file containing the information about the boulders
        projection_center  (Tuple[float, float, float]): projection center if project_boulders is True
        offset_ratio (float): 0.0 => center on the surface, 1.0 means a sphere would be at the limit of penetrating the surface
        labels (int): 255 => label of the rocks on the label map

    """

    project_boulders: bool = True  # use LoadAndProjectMultipleMeshes need a projection center
    attach_objects: bool = False  # to use if the main object moves
    scale_factor: float = 1.0  # to grow the size of objects
    brdf: str = "hapke.brdf"
    brdf_args: Optional[Dict] = field(default_factory=dict)
    coordinates_file_path : str = ""
    projection_center: Tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 1.0)
    )  #  Center of projection if LoadAndProjectMultipleMeshes is used
    offset_ratio: float = 0.3  # offset_ratio of 0 means the center is on the surface, 1 means a sphere would be at the limit of penetrating the surface.
    labels: int = 255


@dataclass_json
@dataclass
class DIGConfig:
    """ Top Level Config, contains everything

    Args:
        scene_config (SceneConfig): scene config
        trajectory_config (TrajectoryConfig): trajectory config
        server_config (ServerConfig): surrender server hostname, port, timeout...
        simulation_config (SimulationConfig): number of rays per pixels, ...
        boulder_config (BoulderConfig): boulder specific config
        output_config (OutputConfig): output folder path, ...
        other_config (Dict): everything else
    """

    scene_config: SceneConfig = field(default_factory=lambda: SceneConfig())
    trajectory_config: TrajectoryConfig = field(
        default_factory=lambda: TrajectoryConfig()
    )
    server_config: ServerConfig = field(default_factory=lambda: ServerConfig())
    simulation_config: SimulationConfig = field(
        default_factory=lambda: SimulationConfig()
    )
    boulder_config: BoulderConfig = field(default_factory=lambda: BoulderConfig())
    output_config: OutputConfig = field(default_factory=lambda: OutputConfig())
    other_config: Dict = field(default_factory=dict)

    @classmethod
    def load_from_text(cls, text: str):
        """ Load DIGConfig from JSON text """
        config = cls.from_json(text)
        return config

    @classmethod
    def load_from_file(cls, path: str):
        """ Load DIGConfig from JSON file """
        with open(path, "r") as json_file:
            config = cls.from_json(json_file.read())
            return config

    @staticmethod
    def create_template():
        """ Create default DIGConfig """
        template = DIGConfig()
        return template.to_json()


def test_dig_config():
    """ check default config serialization/deserialization """
    # check template is a valid config
    template_json = DIGConfig.create_template()
    config_from_template = DIGConfig.load_from_text(template_json)
    assert config_from_template is not None


if __name__ == "__main__":
    print(DIGConfig.create_template())
