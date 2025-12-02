""" 
Lunar Surface Image Generator
==============================

This script generates synthetic lunar surface images along a predefined trajectory
using the SurRender rendering engine.

It produces:
- Rendered grayscale images of the lunar surface (PNG format)
- Associated metadata files (NPZ format) containing:
  - Depth maps
  - Label maps (semantic segmentation)
  - Line-of-sight (LOS) maps
  - Camera intrinsic matrix (K)
  - Camera extrinsic parameters (R_w2c, t_w2c)

The script reads camera poses from a trajectory CSV file and sun positions
from a separate CSV file to simulate realistic lighting conditions.

Usage:
------
    python generate_images.py MR --host localhost --port 5151
    python generate_images.py MR --skip_metadata --image_step 10

Requirements:
-------------
- SurRender server running and accessible
- DEM file (Digital Elevation Model) for lunar surface
- Trajectory CSV file with camera positions and orientations
- Sun trajectory CSV file with sun positions

Output Structure:
-----------------
    outputs/
    └── <scenario_name>/
        ├── images/
        │   ├── im_00000.png
        │   ├── im_00001.png
        │   └── ...
        └── metadata/
            ├── im_00000.npz
            ├── im_00001.npz
            └── ...
Author: Clémentine GRETHEN


"""

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation

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


# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

def get_scenarii_dict() -> Dict[str, Tuple[SceneConfig, TrajectoryConfig]]:
    """
    Define available simulation scenarios.
    
    Each scenario consists of:
    - SceneConfig: DEM file path, BRDF model for surface reflectance
    - TrajectoryConfig: camera trajectory file, sun trajectory file
    
    Returns:
    --------
    Dict mapping scenario names to (SceneConfig, TrajectoryConfig) tuples
    """
    SCENARII = dict(
        MR=(  # Medium Resolution scenario
            SceneConfig(
                DEM_path="YOURPATHTOLUNARDEM.dem",  # Path to Digital Elevation Model
                body_brdf="hapke.brdf",              # Hapke BRDF for lunar surface
            ),
            TrajectoryConfig(
                trajectory_path="/path/to/trajectory.csv",      # Camera trajectory
                sun_trajectory_path="/path/to/sun_trajectory.csv",  # Sun positions
            ),
        ),
    )
    return SCENARII


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_folders(scenarii: List[str]):
    """
    Create output directory structure for each scenario.
    
    Creates:
        outputs/<scenario>/images/   - for rendered PNG images
        outputs/<scenario>/metadata/ - for NPZ metadata files
    """
    for sc in scenarii:
        for folder in ["images", "metadata"]:
            os.makedirs(os.path.join("outputs", sc, folder), exist_ok=True)




# =============================================================================
# RENDERING FUNCTIONS
# =============================================================================

def render_metadata(
    sc_name: str,
    scene_config: SceneConfig,
    trajectory_config: TrajectoryConfig,
    server_config: ServerConfig,
    metadata_step: int,
    logger,
):
    """
    Generate metadata for each frame along the trajectory.
    
    Metadata includes:
    - label_map: Semantic segmentation (body=255, background=0)
    - dmap: Depth map (distance from camera to surface in meters)
    - los_map: Line-of-sight direction vectors
    - K: 3x3 camera intrinsic matrix
    - R_w2c: 3x3 world-to-camera rotation matrix
    - t_w2c: 3x1 world-to-camera translation vector
    
    Parameters:
    -----------
    sc_name : str
        Scenario name (used for output directory)
    scene_config : SceneConfig
        Scene configuration (DEM, BRDF)
    trajectory_config : TrajectoryConfig
        Trajectory file paths
    server_config : ServerConfig
        SurRender server settings
    metadata_step : int
        Generate metadata every N frames (1 = all frames)
    logger : Logger
        Logging instance
    """
    # Create renderer with minimal ray count (metadata doesn't need anti-aliasing)
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

    # Enable metadata outputs in SurRender
    renderer_4_meta.client.enableLabelMapping(True)
    renderer_4_meta.client.enableLOSmapping(True)
    renderer_4_meta.client.setObjectLabel("body", 255)  # Label lunar surface as 255
    
    # Compute camera intrinsic matrix K from image size and field of view
    w, h = renderer_4_meta.config.simulation_config.image_shape
    fov_x, fov_y = renderer_4_meta.config.simulation_config.cam_fov_deg
    fx = w / (2 * np.tan(np.deg2rad(fov_x / 2)))  # Focal length in pixels (x)
    fy = h / (2 * np.tan(np.deg2rad(fov_y / 2)))  # Focal length in pixels (y)
    cx = w / 2  # Principal point x
    cy = h / 2  # Principal point y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    def compute_extrinsics(pos: np.ndarray, quat_wxyz: Tuple[float, float, float, float]):
        """
        Compute world-to-camera extrinsic parameters from position and quaternion.
        
        Parameters:
        -----------
        pos : np.ndarray
            Camera position in world coordinates [x, y, z]
        quat_wxyz : Tuple
            Camera orientation as quaternion (w, x, y, z)
            
        Returns:
        --------
        R_w2c : np.ndarray
            3x3 world-to-camera rotation matrix
        t_w2c : np.ndarray  
            3x1 world-to-camera translation vector
        """
        # Convert quaternion (w,x,y,z) to scipy format (x,y,z,w)
        q = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R_c2w = q.as_matrix()   # Camera-to-world rotation
        R_w2c = R_c2w.T         # World-to-camera rotation (transpose)
        t_w2c = -R_w2c @ pos    # World-to-camera translation
        return R_w2c.astype(np.float32), t_w2c.astype(np.float32)
    
    # Process trajectory frames
    idx_range = np.arange(0, renderer_4_meta.cam_traj.get_length(), metadata_step)
    for img_idx in tqdm(idx_range, desc="Generating metadata"):
        logger.debug(f"computing label map, depthmap and los map for {img_idx=}")
        renderer_4_meta.render_frame(img_idx)

        client = renderer_4_meta.client
        
        # Retrieve maps from SurRender
        label_map = client.getLabelMap()
        los_map = client.getLOSMap()
        dmap = client.getDepthMap()
        
        # Get camera pose
        pos = np.array(client.getObjectPosition("camera"), dtype=np.float32)
        quat_wxyz = tuple(client.getObjectAttitude("camera"))  # (w, x, y, z)
        
        # Compute world-to-camera transformation
        q = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R_c2w = q.as_matrix()
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ pos


        # Validate retrieved data
        assert label_map is not None and los_map is not None and dmap is not None

        # Save metadata to compressed NPZ file
        logs = {
            "label_map": label_map.astype(np.uint8),   # Semantic labels
            "dmap": dmap.astype(np.float32),           # Depth map
            "los_map": los_map.astype(np.float32),     # Line-of-sight vectors
            "K": K,                                      # Intrinsic matrix
            "R_w2c": R_w2c,                              # Rotation world->camera
            "t_w2c": t_w2c,                              # Translation world->camera
        }
        np.savez_compressed(f"outputs/{sc_name}/metadata/im_{img_idx:05d}.npz", **logs)


def render_images(
    sc_name: str, renderer: TrajectoryRenderer, image_step: int, logger,
):
    """
    Render grayscale images along the trajectory.
    
    Images are rendered as 32-bit float, then normalized to 8-bit PNG.
    
    Parameters:
    -----------
    sc_name : str
        Scenario name (used for output directory)
    renderer : TrajectoryRenderer
        Configured trajectory renderer with full ray count
    image_step : int
        Render every N frames (1 = all frames)
    logger : Logger
        Logging instance
    """
    idx_range = np.arange(0, renderer.cam_traj.get_length(), image_step)
    
    for img_idx in tqdm(idx_range, desc="Rendering images"):
        logger.debug(f"rendering {img_idx=}")
        renderer.render_frame(img_idx)
        
        # Get rendered image as 32-bit float grayscale
        img = renderer.get_visible_frame(img_format=ImageFormat.Gray32F)
        logger.debug(f"{img.min()=} {img.max()=}")

        # Normalize to 8-bit range [0, 255] for PNG output
        img8b = (
            255 * np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)
        ).astype(np.uint8)

        Image.fromarray(img8b).save(f"outputs/{sc_name}/images/im_{img_idx:05d}.png")
    


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def get_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_images.py",
        description="Generate lunar surface images using SurRender",
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
        "--image_rays",
        type=int,
        default=64,
        help="number of rays when simulating images",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="SurRender server host or ip"
    )
    parser.add_argument("--port", type=int, default=5151, help="SurRender server port")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="SurRender timeout (s)"
    )
    return parser


# =============================================================================
# MAIN SIMULATION
# =============================================================================

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
):
    """
    Run the complete image generation simulation.
    
    Parameters:
    -----------
    scenarii : List[str]
        List of scenario names to process
    surrender_host : str
        SurRender server hostname or IP
    surrender_port : int
        SurRender server port
    surrender_timeout : float
        Connection timeout in seconds
    metadata_step : int
        Generate metadata every N frames
    skip_metadata : bool
        Skip metadata generation if True
    image_step : int
        Render images every N frames
    image_rays : int
        Rays per pixel for image rendering (quality)
    skip_image : bool
        Skip image rendering if True
    """
    SCENARII = get_scenarii_dict()

    create_output_folders(scenarii)

    logger = get_logger("generate_images.py")

    # Configure SurRender server connection
    server_config = ServerConfig(
        hostname=surrender_host,
        port=surrender_port,
        resource_path="/path/to/surrender/resources",  # Path to SurRender data files
        timeout=surrender_timeout,
    )
    
    logger.debug(f"{server_config=}")
    logger.debug(f"the following scenarii will be run: {scenarii}")
    
    # Process each scenario
    for sc_name in scenarii:
        scene_config, trajectory_config = SCENARII[sc_name]
        logger.debug(f"running {sc_name}")

        # Create renderer for image generation (with full ray count for quality)
        renderer = TrajectoryRenderer(
            config=DIGConfig(
                scene_config=scene_config,
                trajectory_config=trajectory_config,
                server_config=server_config,
                simulation_config=SimulationConfig(
                    rays_per_pixels=image_rays,       # Higher = better quality
                    image_shape=(512, 512),           # Output resolution
                    cam_fov_deg=(45.0, 45.0),         # Field of view (degrees)
                    psf_name="gaussian",              # Point spread function
                    psf_args=dict(sigma=0.6),         # PSF blur parameter
                ),
                output_config=OutputConfig(),
            )
        )

        # Generate metadata (depth maps, labels, camera params)
        if not skip_metadata:
            render_metadata(
                sc_name, scene_config, trajectory_config, server_config, metadata_step, logger,
            )
        
        # Generate rendered images
        if not skip_image:
            render_images(sc_name, renderer, image_step, logger)


# =============================================================================
# ENTRY POINT
# =============================================================================

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
    )
