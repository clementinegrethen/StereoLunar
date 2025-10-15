import cv2
import json
import os
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type, Union, List, Tuple, Dict, Optional

import numpy as np
import pandas
from PIL import Image
from numpy import linalg
from tqdm import tqdm

from descentimagegenerator.config import (
    DIGConfig,
    SceneConfig,
    ServerConfig,
    SimulationConfig,
    OutputConfig,
    TrajectoryConfig,
)
from descentimagegenerator.trajectoryreader import Trajectory
from descentimagegenerator.trajectoryrenderer import TrajectoryRenderer, ImageFormat
from descentimagegenerator.logger import get_logger
from surrender.geometry import normalize, vec3, look_at


def get_scenarii_dict() -> Dict[str, Tuple[SceneConfig, TrajectoryConfig]]:
    SCENARII = dict(
        MR=(
            SceneConfig(DEM_path="south5m.dem", body_brdf="hapke.brdf",body_brdf_args={"albedo": [0.12, 0.12, 0.12, 0.12]},),
            TrajectoryConfig(
                trajectory_path="/home/cgrethen/Documents/3Dreconstruction/Dataset/Moon/SurrenderScript/traj_landing_test.csv",
                sun_trajectory_path="/home/cgrethen/Documents/3Dreconstruction/Dataset/Moon/SurrenderScript/sun_landing_test.csv",
            ),
        ),
        HR=(
            SceneConfig(
                DEM_path="change2_20m_fused.dem",
                body_brdf="hapke.brdf",
                hybrid_textures_lua_path=os.path.join(
                    os.getcwd(), "conf/CE3_demonly.lua"
                ),
            ),
            TrajectoryConfig(
                trajectory_path="data/ce3-traj.npz",
                sun_trajectory_path="data/ce3-traj.npz",
            ),
        ),
        HR_blending=(
            SceneConfig(
                DEM_path="change2_20m_fused.dem",
                body_brdf="hapke.brdf",
                hybrid_textures_lua_path=os.path.join(
                    os.getcwd(), "conf/CE3_dem_blending.lua"
                ),
            ),
            TrajectoryConfig(
                trajectory_path="data/ce3-traj.npz",
                sun_trajectory_path="data/ce3-traj.npz",
            ),
        ),
        HR_proc=(
            SceneConfig(
                DEM_path="change2_20m_fused.dem",
                body_brdf="hapke.brdf",
                hybrid_textures_lua_path=os.path.join(
                    os.getcwd(), "conf/CE3_dem_blending_suwog_nol.lua"
                ),
            ),
            TrajectoryConfig(
                trajectory_path="data/ce3-traj.npz",
                sun_trajectory_path="data/ce3-traj.npz",
            ),
        ),
    )

    for fn in list(Path("data").glob("*.mat")):
        SCENARII[f"{fn.stem}-HR"] = (
            SceneConfig(
                DEM_path="south5m.dem",
                body_brdf="hapke.brdf",
                hybrid_textures_lua_path=os.path.join(
                    os.getcwd(), "conf/CE3_demonly.lua"
                ),
            ),
            TrajectoryConfig(
                trajectory_path=str(fn),
                sun_trajectory_path=str(fn),
                sun_trajectory_config={"sun_index": 3800},
            ),
        )
    return SCENARII


def create_output_folders(scenarii: List[str]):
    for sc in scenarii:
        for folder in ["imageLandindingTest", "metadataLandingTest"]:
            os.makedirs(os.path.join("outputs", sc, folder), exist_ok=True)


def compute_up_for_trajectory(
    trajectory: Trajectory, idx_range,
) -> Tuple[float, float, float]:
    return np.mean([trajectory.get_position(idx) for idx in idx_range], axis=0)


def render_and_fix_matlab_missing_camera_attitude_if_necessary(
    sc_name: str,
    img_idx: int,
    renderer: TrajectoryRenderer,
    up: Optional[Tuple[float, float, float]],
    logger,
):
    necessary = sc_name in (
        "2023-07-13_ChangE_fast-HR",
        "2023-07-13_ChangE_moy-HR",
        "2023-07-13_ChangE_slow-HR",
    )
    if necessary:
        logger.debug("using look_at for the lack of quaternions in traj file")
        renderer.update_objects_positions(img_idx)

        eye_pos = renderer.cam_traj.get_position(img_idx)
        target_pos = renderer.cam_traj.get_position(renderer.cam_traj.get_length() - 1)
        assert up is not None
        logger.debug(f"look_at({eye_pos=}, {target_pos=}, {up=})")
        if np.linalg.norm(np.array(eye_pos) - np.array(target_pos)) < 1e-3:
            target_pos = (
                target_pos * 0.9
            )  # fix end of trajectory where eye_pos is too close to target_pos
            logger.debug(f"look_at({eye_pos=}, {target_pos=}, {up=})")
        look_at(renderer.client, eye_pos=eye_pos, target_pos=target_pos, up=up)

        if renderer.use_sensor:
            renderer.client.runLuaCode("sensor:render();")
        else:
            renderer.client.render()
    else:
        renderer.render_frame(img_idx)
        
from scipy.spatial.transform import Rotation

def render_metadata(
    sc_name: str,
    scene_config: SceneConfig,
    trajectory_config: TrajectoryConfig,
    server_config: ServerConfig,
    metadata_step: int,
    logger,
):
    renderer_4_meta = TrajectoryRenderer(
        config=DIGConfig(
            scene_config=scene_config,
            trajectory_config=trajectory_config,
            server_config=server_config,
            simulation_config=SimulationConfig(
                rays_per_pixels=1,  # must remain 1 when generating metadata
                image_shape=(512, 512),
                cam_fov_deg=(45,45),
            ),
            output_config=OutputConfig(),
        )
    )

    renderer_4_meta.client.enableLabelMapping(True)
    renderer_4_meta.client.enableLOSmapping(True)

    renderer_4_meta.client.setObjectLabel("body", 255)
 # Compute intrinsics manually
    w, h      = renderer_4_meta.config.simulation_config.image_shape
    fov_x, fov_y = renderer_4_meta.config.simulation_config.cam_fov_deg
    fx = w / (2 * np.tan(np.deg2rad(fov_x / 2)))
    fy = h / (2 * np.tan(np.deg2rad(fov_y / 2)))
    cx = w / 2
    cy = h / 2
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    def compute_extrinsics(pos: np.ndarray, quat_wxyz: Tuple[float, float, float, float]):
            # Build rotation world->camera from quaternion
            # quat_wxyz: (w, x, y, z)
            q = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            R_c2w = q.as_matrix()          # camera to world
            R_w2c = R_c2w.T                # world to camera
            t_w2c = -R_w2c @ pos           # translation world->camera
            return R_w2c.astype(np.float32), t_w2c.astype(np.float32)
    idx_range = np.arange(0, renderer_4_meta.cam_traj.get_length(), metadata_step)
    up = compute_up_for_trajectory(renderer_4_meta.cam_traj, idx_range)
    for img_idx in tqdm(idx_range):
        logger.debug(f"computing label map, depthmap and los map for {img_idx=}")
        render_and_fix_matlab_missing_camera_attitude_if_necessary(
            sc_name, img_idx, renderer_4_meta, up, logger
        )

        client = renderer_4_meta.client
        label_map = client.getLabelMap()
        los_map = client.getLOSMap()
        dmap = client.getDepthMap()
        pos = np.array(client.getObjectPosition("camera"), dtype=np.float32)  # [x, y, z]
        quat_wxyz = tuple(client.getObjectAttitude("camera"))  # (w, x, y, z)
        q = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R_c2w = q.as_matrix()
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ pos
        #save camera parameters: intrisics and world--to-camera matrix


        assert label_map is not None and los_map is not None and dmap is not None

        logs = dict()

        logs["label_map"] = label_map.astype(np.uint8)
        logs["dmap"] = dmap.astype(np.float32)
        logs["los_map"] = los_map.astype(np.float32)
        logs["K"] = K
        logs["R_w2c"] = R_w2c
        logs["t_w2c"] = t_w2c

        np.savez_compressed(f"outputs/{sc_name}/metadataLandingTest/im_{img_idx:05d}.npz", **logs)


def render_images(
    sc_name: str, renderer: TrajectoryRenderer, image_step: int, logger,
):
    idx_range = np.arange(0, renderer.cam_traj.get_length(), image_step)
    up = compute_up_for_trajectory(renderer.cam_traj, idx_range)
    for img_idx in tqdm(idx_range):
        logger.debug(f"rendering {img_idx=}")
        render_and_fix_matlab_missing_camera_attitude_if_necessary(
            sc_name, img_idx, renderer, up, logger
        )
        img = renderer.get_visible_frame(img_format=ImageFormat.Gray32F)
        logger.debug(f"{img.min()=} {img.max()=}")

        # normalize image for 8 bit PNG
        img8b = (
            255 * np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)
        ).astype(np.uint8)

        Image.fromarray(img8b).save(f"outputs/{sc_name}/imageLandindingTest/im_{img_idx:05d}.png")
    

def rock_stuff(
    renderer: TrajectoryRenderer, logger,
):

    T = TypeVar("T")

    def load(target_type: Type[T], data: dict) -> T:
        return target_type(**data)

    def load_json(target_type: Type[T], path: Union[Path, str]):
        with Path(path).open() as _:
            data = json.load(_)
            return load(target_type, data)

    def read_csv_config(filename: str):
        df = pandas.read_csv(filename)
        dict_ = df.to_dict("list")
        for k in dict_.keys():
            dict_[k] = [it for it in dict_[k] if not (pandas.isnull(it)) == True]
        return dict_

    @dataclass
    class ItemDEMRes:
        model_id: int = 0
        size: float = 1.0
        position_x: float = 0.0
        position_y: float = 0.0
        position_z: float = 0.0

    @dataclass
    class ItemDEMTemplateRes:
        name: str
        items: List[ItemDEMRes]
        image_size: Tuple[int, int] = None
        image_resolution: Tuple[float, float] = None

    @dataclass
    class ItemDEMLonLat:
        model_id: int = 0
        size: float = 1.0
        lon: float = 0
        lat: float = 0

    @dataclass
    class ItemDEMLonLatTemplate:
        name: str
        items: List[ItemDEMLonLat]
        image_size: Tuple[int, int] = None
        image_resolution: Tuple[float, float] = None

    def pos_to_lonlat(pos, Rmoon):
        # vecteur normal a la surface positionné en G
        n_Pos = pos / linalg.norm(pos)
        G = n_Pos * Rmoon

        x = G[0]
        y = G[1]
        z = G[2]

        lon = np.arctan2(y, x)
        lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

        return lon, lat

    def lonlat_to_pos(lon_deg, lat_deg, Rmoon):

        lon = lon_deg * np.pi / 180
        lat = lat_deg * np.pi / 180

        x = np.cos(lon) * np.cos(lat)
        y = np.sin(lon) * np.cos(lat)
        z = np.sin(lat)

        pos = np.array([x, y, z]) * (Rmoon)

        return pos

    MOON_RADIUS_M = 1_737_400

    ROCK_BDD = "/home/nollagnier/shackleton_ssd0/amasson/CE3/scripts"
    scene_json_path = "/home/nollagnier/shackleton_ssd0/ybp"
    scene_rocks_name = "rocks_scene"

    rockJsonFileLonLat = os.path.join(
        scene_json_path, f"{scene_rocks_name}_lonlat_items.json"
    )
    model_meshes_d = read_csv_config(os.path.join(ROCK_BDD, f"rock_config.csv"))
    items_scene = load_json(ItemDEMLonLatTemplate, rockJsonFileLonLat)

    # x_size=17869.99999999964
    # y_size=76644.99999999846

    MINIMUM_LATITUDE = 43.0997888
    MAXIMUM_LATITUDE = 45.6273789
    EASTERNMOST_LONGITUDE = 341.0332536
    WESTERNMOST_LONGITUDE = 340.2140095
    # MIDDLE_LON = (EASTERNMOST_LONGITUDE + WESTERNMOST_LONGITUDE)/2. #MIDDLE_LAT = (MINIMUM_LATITUDE + MAXIMUM_LATITUDE) /2.
    # ECART_LON = (EASTERNMOST_LONGITUDE - WESTERNMOST_LONGITUDE) #ECART_LAT = (MAXIMUM_LATITUDE - MINIMUM_LATITUDE )
    # MINIMUM_LATITUDE= MIDDLE_LAT-0.89*ECART_LAT #MAXIMUM_LATITUDE= MIDDLE_LAT+1.*ECART_LAT
    # EASTERNMOST_LONGITUDE= MIDDLE_LON#+0.5*ECART_LON #WESTERNMOST_LONGITUDE= MIDDLE_LON-0.45*ECART_LON

    logger.info("Load and project rock meshes")

    # scene_rocks_name = 'rock_scene'
    # rockJsonFile = os.path.join(scene_json_path, f'{scene_rocks_name}_items.json')
    # rockJsonFile = "rocks_scene_items.json"
    # items_scene = load_json(ItemDEMTemplateRes, rockJsonFile)
    # model_meshes_d = read_csv_config("rock_config.csv")
    meshes, positions, attitudes = list(), list(), list()
    offset = MAXIMUM_LATITUDE - MINIMUM_LATITUDE

    i = 0
    for item_ in items_scene.items:
        i += 1
        meshes.append(str(random.choice(model_meshes_d[str(item_["model_id"])])))
        lon, lat = item_["lon"], item_["lat"]
        if lon > EASTERNMOST_LONGITUDE or lon < WESTERNMOST_LONGITUDE:
            print(lon)
        if lat > MAXIMUM_LATITUDE or lon < MINIMUM_LATITUDE:
            print(lon)
        # lon, lat = 340.6, MINIMUM_LATITUDE #- offset
        pos = lonlat_to_pos(
            lon, lat + offset, MOON_RADIUS_M + 10_000
        )  # -2585) #0.9987  #* 0.9985 +
        positions.append(pos)
        attitudes.append(
            (
                1e-1 * item_["size"] * normalize(np.random.randn(4)).astype(np.float64)
            ).tolist()
        )

        # if i == 20:  # 15000:
        #     break

    meshes = ["Rock01_OBJ_nol.obj" for _m in meshes]

    # set sun position *before* attempting to load and project multiple meshes
    pos_sun = vec3(1, -1, -2) * 1e22
    renderer.client.setObjectPosition("sun", pos_sun)

    # renderer.client.setRessourcePath(scene_json_path)
    proj_dir = np.array([0.0, 0.0, 0.0, 1.0])
    renderer.client.loadAndProjectMultipleMeshes(
        scene_rocks_name, meshes, positions, attitudes, proj_dir, 0.0
    )
    # renderer.client.loadMultipleMeshes(scene_rocks_name, meshes, positions, attitudes)
    renderer.client.setObjectElementBRDF(
        scene_rocks_name, scene_rocks_name, "body_brdf"
    )  # 'oren_nayar_moon')
    renderer.client.setObjectProperty(scene_rocks_name, "not_a_secondary_source", True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moon_landing_td_ce3.py",
        description="run moon landing scenarii",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scenarii",
        choices=list(get_scenarii_dict().keys()),
        nargs="+",
        help="scenarii to run",
    )
    parser.add_argument(
        "--image_step",
        type=int,
        default=1,
        help="step for image simulation 1 => everything, 10 => every 10 images",
    )
    parser.add_argument(
        "--metadata_step",
        type=int,
        default=1,
        help="step for image simulation 1 => everything, 10 => every 10 images",
    )
    parser.add_argument(
        "--skip_image", action="store_true", help="skip image generation"
    )
    parser.add_argument(
        "--skip_metadata", action="store_true", help="skip metadata generation"
    )
    parser.add_argument(
        "--use_rocks", action="store_true", help="add rock meshes (slower)"
    )
    parser.add_argument(
        "--image_rays",
        type=int,
        default=64,
        help="number of rays when simulating images",
    )
    parser.add_argument(
        "--host", type=str, default="shackleton", help="SurRender server host or ip"
    )
    parser.add_argument("--port", type=int, default=5113, help="SurRender server port")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="SurRender timeout (s)"
    )
    return parser


def run_simulations(
    scenarii: List[str],
    surrender_host: str,
    surrender_port: int,
    surrender_timeout: float,
    metadata_step: int,
    skip_metadata: bool,
    image_step: int,
    image_rays: int,
    skip_image: bool,
    use_rocks: bool,
):
    SCENARII = get_scenarii_dict()

    create_output_folders(scenarii)

    logger = get_logger("moon_landing_td_ce3.py")

    # connect to SurRender
    server_config = ServerConfig(
        hostname=surrender_host,
        port=surrender_port,
        resource_path="/mnt/ssd0/nol/surrender_data",
        timeout=surrender_timeout,
    )
    logger.debug(f"{server_config=}")
    logger.debug(f"the following scenarii will be run: {scenarii}")
    # {"MR", "HR", "2023-07-13_ChangE_fast-HR", "2023-07-13_ChangE_moy-HR", "2023-07-13_ChangE_slow-HR", "HR_proc"}
    for sc_name in scenarii:
        scene_config, trajectory_config = SCENARII[sc_name]
        logger.debug(f"running {sc_name}")

        renderer = TrajectoryRenderer(
            config=DIGConfig(
                scene_config=scene_config,
                trajectory_config=trajectory_config,
                server_config=server_config,
                simulation_config=SimulationConfig(
                    rays_per_pixels=image_rays,
                    image_shape=(512, 512),
                    cam_fov_deg=(45.0, 45.0),
                    psf_name="gaussian",
                    psf_args=dict(sigma=0.6),
                ),
                output_config=OutputConfig(),
            )
        )

        logger.debug(f"{use_rocks=}")

        if use_rocks:
            rock_stuff(renderer, logger)

        if not skip_metadata:
            render_metadata(
                sc_name, scene_config, trajectory_config, server_config, metadata_step, logger,
            )
        if not skip_image:
            render_images(sc_name, renderer, image_step, logger)


if __name__ == "__main__":
    CLI_ARGS = get_parser().parse_args()
    run_simulations(
        scenarii=CLI_ARGS.scenarii,
        surrender_host=CLI_ARGS.host,
        surrender_port=CLI_ARGS.port,
        surrender_timeout=CLI_ARGS.timeout,
        metadata_step=CLI_ARGS.metadata_step,
        skip_metadata=CLI_ARGS.skip_metadata,
        image_step=CLI_ARGS.image_step,
        image_rays=CLI_ARGS.image_rays,
        skip_image=CLI_ARGS.skip_image,
        use_rocks=CLI_ARGS.use_rocks,
    )
