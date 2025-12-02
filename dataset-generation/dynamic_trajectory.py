#!/usr/bin/env python3
"""
Dynamic Stereo Pair Generator for Lunar Surface
================================================
This script generates HIGHLY DYNAMIC stereo image pairs of the lunar surface
using the SurRender rendering engine.

KEY DIFFERENCE FROM NADIR/OBLIQUE SCRIPTS:
------------------------------------------
Unlike nadir_trajectory.py (cameras pointing straight down) or 
oblique_trajectory.py (cameras at fixed angles), THIS script creates
stereo pairs with VARIED and UNPREDICTABLE viewing geometries:

- Both cameras can have DIFFERENT orientations (not symmetric)
- Each camera independently varies its altitude (±30%)
- Viewing directions can deviate significantly from target
- Roll variations add rotational complexity

This simulates realistic scenarios where camera poses are not perfectly
controlled, creating challenging stereo pairs for robust algorithms.

DYNAMIC INTENSITY LEVELS:
-------------------------
All modes are MORE DYNAMIC than nadir/oblique. The levels control intensity:

1. "mild_dynamic" (30% probability):
   - Pitch variation: ±5° from target direction
   - Entry-level dynamic: still more varied than nadir/oblique
   - Good for algorithms transitioning from classical stereo

2. "moderate_dynamic" (40% probability):
   - Pitch variation: ±10° from target direction  
   - Standard dynamic complexity
   - Tests stereo matching robustness

3. "extreme_dynamic" (30% probability):
   - Pitch variation: ±15° from target direction
   - Maximum geometric diversity
   - Stress-tests stereo reconstruction algorithms

WHAT MAKES THIS "DYNAMIC":
--------------------------
- ASYMMETRIC stereo: cameras can have completely different configurations
- INDEPENDENT altitude: each camera varies ±30% from base (different heights)
- VARIABLE pointing: up to ±30° deviation from target direction
- ROLL diversity: rotational variations between cameras
- B/H ratio maintains baseline proportional to altitude

COMPARISON WITH OTHER SCRIPTS:
------------------------------
| Script   | Geometry       | Camera Symmetry | Altitude Variation |
|----------|----------------|-----------------|--------------------|
| nadir    | Vertical down  | Symmetric       | Fixed per level    |
| oblique  | Fixed angles   | Symmetric       | Strategy-based     |
| dynamic  | VARIABLE       | ASYMMETRIC      | ±30% independent   |

VIEWING CONSTRAINT LEVELS:
--------------------------
- "tight": ±10° max deviation (mildest dynamic)
- "moderate": ±20° max deviation (standard dynamic)
- "loose": ±30° max deviation (extreme dynamic)

Requirements:
    - SurRender rendering engine with Python client
    - NumPy, Pandas, SciPy
    - A lunar DEM file (e.g., LOLA data)

Author: Clémentine GRETHEN
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4

# ════════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
MOON_RADIUS = 1_737_400  # Moon radius in meters

# Define altitude levels and their distribution
ALTITUDE_DISTRIBUTION = [
    {"altitude": 4000,  "pairs": 1},   # 10%
    {"altitude": 7000,  "pairs": 1},   # 10%
    {"altitude": 10000,  "pairs": 1},  # 10%
    {"altitude": 13000, "pairs": 1},   # 10%
    {"altitude": 16000, "pairs": 1},   # 10%
    {"altitude": 19000, "pairs": 1},   # 10%
    {"altitude": 22000, "pairs": 1},   # 10%
    {"altitude": 25000, "pairs": 1},   # 10%
    {"altitude": 28000, "pairs": 1},   # 10%
    {"altitude": 31000, "pairs": 1},   # 10%
]

# Total pairs is automatically computed from the altitude distribution
TOTAL_PAIRS = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)

# ════════════════════════════════════════════════════════════════════════════════
# DYNAMIC STEREO CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
# ALL modes are more dynamic than nadir/oblique scripts.
# The three levels represent INTENSITY of dynamic behavior:
#   - mild_dynamic: entry-level dynamic (still more varied than nadir/oblique)
#   - moderate_dynamic: standard dynamic complexity
#   - extreme_dynamic: maximum geometric challenge

CAMERA_MODES = ["mild_dynamic", "moderate_dynamic", "extreme_dynamic"]

# Pitch adjustment for each mode (added on top of target-pointing direction)
# All ranges are MORE VARIED than nadir (0°) or oblique (fixed angle)
MILD_DYNAMIC_PITCH_RANGE = (-5, 5)       # Entry-level dynamic
MODERATE_DYNAMIC_PITCH_RANGE = (-10, 10) # Standard dynamic
EXTREME_DYNAMIC_PITCH_RANGE = (-15, 15)  # Maximum challenge

# Altitude variation for each camera (independently)
# Each camera varies independently for maximum diversity
ALT_VARIATION_RANGE = (-0.3, 0.3)  # ±30% possible variation

# Roll variation range - adds rotational diversity
ROLL_VARIATION_RANGE = (-10, 10)  # ±10° roll

# ════════════════════════════════════════════════════════════════════════════════
# BASELINE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
# B/H ratio (baseline-to-height) maintains consistent stereo angles across altitudes
B_H_RATIO_MIN = 0.02  # Minimum B/H ratio (2%)
B_H_RATIO_MAX = 0.22  # Maximum B/H ratio (22%)

# DEM coverage area (customize for your target region)
LAT_RANGE = (-89.5, -87.6)  # Latitude range - south pole region
LON_RANGE = (0.0, 360)      # Full longitude coverage

# Grid for uniform spatial coverage
N_LAT_BINS = 40  # Number of latitude bands
N_LON_BINS = 60  # Number of longitude sectors

# Parameters for infinity pixel filtering
MAX_INFINITY_PIXELS = 50            # Maximum infinity pixels accepted
DEPTH_CHECK_RESOLUTION = 128        # Resolution for quick depth test
INFINITY_THRESHOLD = 100_000_00     # Beyond 100km considered as infinity

N_LIGHTS = 3
AU = 149_597_870_700  # Astronomical Unit in meters

def generate_sun_setups():
    """Generate sun lighting configurations (azimuth, incidence) in degrees."""
    setups = [
        (150, 160),  # Low sun angle
        (250, 20),   # Different azimuth
        (360, 165)   # High incidence
    ]
    return setups

SUN_SETUPS = generate_sun_setups()
print(f"Sun setups generated: {SUN_SETUPS}")


# ════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    return v if n < 1e-16 else v / n


def frame2quat(forward, right):
    """
    Convert a camera frame (forward, right vectors) to a quaternion.
    
    Returns quaternion in (w, x, y, z) format for SurRender.
    """
    z = normalize(forward)
    x = normalize(right)
    y = normalize(np.cross(z, x))
    R = np.column_stack((x, y, z))
    q = Rotation.from_matrix(R).as_quat()
    return (q[3], q[0], q[1], q[2])


def look_at_quat(eye_pos, target_pos):
    """
    Create a quaternion that makes the camera look at a target position.
    
    Parameters:
    -----------
    eye_pos : array
        Camera position
    target_pos : array
        Target to look at
    
    Returns:
    --------
    tuple : Quaternion (w, x, y, z)
    """
    z = normalize(target_pos - eye_pos)
    radial = normalize(eye_pos)
    east = normalize(np.cross([0, 0, 1], radial))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    x = normalize(np.cross(east, z))
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    q = Rotation.from_matrix(R).as_quat()
    return (q[3], q[0], q[1], q[2])


def geodetic_to_cartesian_BCBF_position(lon, lat, alt, a, b):
    """
    Convert geodetic coordinates to Body-Centered Body-Fixed (BCBF) Cartesian.
    
    Parameters:
    -----------
    lon, lat : float
        Longitude and latitude in degrees
    alt : float
        Altitude above reference ellipsoid in meters
    a, b : float
        Semi-major and semi-minor axes of the reference ellipsoid
    
    Returns:
    --------
    array : [X, Y, Z] position in meters
    """
    λ = np.deg2rad(((lon + 180) % 360) - 180)
    φ = np.deg2rad(lat)
    N = a**2 / np.sqrt(a**2*np.cos(φ)**2 + b**2*np.sin(φ)**2)
    X = (N+alt)*np.cos(φ)*np.cos(λ)
    Y = (N+alt)*np.cos(φ)*np.sin(λ)
    Z = ((b**2/a**2)*N+alt)*np.sin(φ)
    return np.array([X, Y, Z], dtype=np.float64)


def sun_position_local(az_deg, inc_deg, P_surf, dist=AU):
    """
    Calculate sun position based on local azimuth and incidence angles.
    
    Parameters:
    -----------
    az_deg : float
        Azimuth angle in degrees (0=North, 90=East)
    inc_deg : float
        Incidence angle in degrees (0=horizontal, 90=zenith)
    P_surf : array
        Surface point position
    dist : float
        Distance to sun (default: 1 AU)
    
    Returns:
    --------
    array : Sun position in Cartesian coordinates
    """
    az = np.deg2rad(az_deg)
    inc = np.deg2rad(inc_deg)
    up = normalize(P_surf)
    east = normalize(np.cross([0, 0, 1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    north = np.cross(up, east)
    dir_local = (np.cos(inc)*(np.cos(az)*north + np.sin(az)*east)
                 + np.sin(inc)*up)
    return dir_local * dist

# ════════════════════════════════════════════════════════════════════════════════
# SPATIAL GRID GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def create_spatial_grid(lat_range, lon_range, n_lat, n_lon):
    """
    Create a grid of points uniformly distributed on the surface,
    accounting for meridian convergence near the poles.
    
    Parameters:
    -----------
    lat_range : tuple
        (min_lat, max_lat) in degrees
    lon_range : tuple
        (min_lon, max_lon) in degrees
    n_lat : int
        Number of latitude bands
    n_lon : int
        Number of longitude sectors
    
    Returns:
    --------
    list : List of (lat, lon) tuples with small random jitter
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    grid_points = []
    
    for i in range(n_lat):
        lat_center = (lat_edges[i] + lat_edges[i+1]) / 2
        # Reduce longitude points near poles where meridians converge
        reduction_factor = max(0.3, abs(np.cos(np.deg2rad(lat_center))))
        n_lon_adjusted = max(8, int(n_lon * reduction_factor))
        lon_centers = np.linspace(lon_min, lon_max, n_lon_adjusted, endpoint=False)
        
        for lon in lon_centers:
            # Add small jitter for natural distribution
            lat_jitter = np.random.uniform(-0.05, 0.05) * (lat_edges[i+1] - lat_edges[i])
            lon_jitter = np.random.uniform(-0.5, 0.5) * 360 / n_lon_adjusted
            final_lat = np.clip(lat_center + lat_jitter, lat_min, lat_max)
            final_lon = (lon + lon_jitter) % 360
            grid_points.append((final_lat, final_lon))
    
    return grid_points

def check_infinity_in_view(s, position, attitude):
    """
    Check if the view contains pixels at infinity.
    
    This is important for low-altitude scenarios where oblique views
    might see beyond the horizon.
    
    Parameters:
    -----------
    s : surrender_client
        SurRender client instance
    position : array
        Camera position
    attitude : tuple
        Camera quaternion
    
    Returns:
    --------
    tuple : (is_valid, infinity_count)
        - is_valid: True if acceptable number of infinity pixels
        - infinity_count: Number of pixels seeing infinity
    """
    original_size = s.getImageSize()
    s.setImageSize(DEPTH_CHECK_RESOLUTION, DEPTH_CHECK_RESOLUTION)
    s.setObjectPosition("camera", vec3(*position))
    s.setObjectAttitude("camera", attitude)
    s.render()
    depth_map = s.getDepthMap()
    s.setImageSize(original_size[0], original_size[1])
    infinity_mask = (depth_map > INFINITY_THRESHOLD) | (~np.isfinite(depth_map))
    infinity_count = np.sum(infinity_mask)
    return infinity_count <= MAX_INFINITY_PIXELS, infinity_count

def generate_camera_config(P_surf, alt_ground, base_altitude, camera_id, target_point=None, constraint_type="moderate"):
    """
    Generate a DYNAMIC camera configuration with varied geometry.
    
    WHY "DYNAMIC" (vs nadir/oblique):
    ---------------------------------
    Nadir script: Both cameras point straight down, symmetric geometry
    Oblique script: Both cameras at same oblique angle, symmetric geometry
    THIS script: Each camera INDEPENDENTLY configured = ASYMMETRIC geometry
    
    This creates realistic scenarios where:
    - Camera 1 might be at 5000m looking 15° off-target
    - Camera 2 might be at 7000m looking 5° off-target with different roll
    → Challenging stereo pairs that test algorithm robustness
    
    Parameters:
    -----------
    P_surf : array
        Reference surface point position
    alt_ground : float
        Terrain altitude at the surface point
    base_altitude : float
        Base altitude for this level
    camera_id : int
        Camera identifier (1 or 2) - each configured INDEPENDENTLY
    target_point : array, optional
        Target point to view (defaults to P_surf)
    constraint_type : str
        Controls how much cameras can deviate:
        - "tight": ±10° (mild dynamic)
        - "moderate": ±20° (standard dynamic)  
        - "loose": ±30° (extreme dynamic)
    
    Returns:
    --------
    dict : Camera configuration with position, attitude, and metadata
    """
    if target_point is None:
        target_point = P_surf
    
    # Choose dynamic intensity level
    # 30% mild, 40% moderate, 30% extreme - ALL more dynamic than nadir/oblique
    mode = np.random.choice(CAMERA_MODES, p=[0.3, 0.4, 0.3])
    
    # Altitude variation
    alt_variation = np.random.uniform(*ALT_VARIATION_RANGE)
    altitude = base_altitude * (1 + alt_variation)
    
    # Base position
    up = normalize(P_surf)
    
    # Random lateral offset for position
    east = normalize(np.cross([0, 0, 1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    north = np.cross(up, east)
    
    # ─────────────────────────────────────────────────────────────
    # BASELINE PROPORTIONAL TO ALTITUDE
    # Using B/H ratio maintains consistent stereo geometry
    # ─────────────────────────────────────────────────────────────
    b_h_ratio = np.random.uniform(B_H_RATIO_MIN, B_H_RATIO_MAX)
    baseline = altitude * b_h_ratio
    
    # Random direction for baseline
    azimuth = np.random.uniform(0, 360)
    az_rad = np.deg2rad(azimuth)
    
    lateral_offset = baseline * (np.cos(az_rad) * north + np.sin(az_rad) * east)
    
    # Camera position
    P_cam_nominal = P_surf + altitude * up + lateral_offset
    P_cam = normalize(P_cam_nominal) * (MOON_RADIUS + alt_ground + altitude)
    
    # Base direction toward target (ensures common viewing zone)
    forward_to_target = normalize(target_point - P_cam)
    
    # Create local coordinate system
    right = normalize(np.cross(forward_to_target, up))
    if np.linalg.norm(right) < 1e-16:
        right = normalize(np.cross(forward_to_target, north))
    up_local = np.cross(right, forward_to_target)
    
    # Apply limited angular offset based on constraint type
    if constraint_type == "tight":
        max_angle_offset = 10  # Maximum 10° offset from target
    elif constraint_type == "moderate":
        max_angle_offset = 20  # Maximum 20° offset
    else:  # loose
        max_angle_offset = 30  # Maximum 30° offset
    
    # Random angular offset in cone around target direction
    angle_offset = np.random.uniform(0, max_angle_offset)
    angle_azimuth = np.random.uniform(0, 360)
    
    # Convert to radians
    angle_offset_rad = np.deg2rad(angle_offset)
    angle_azimuth_rad = np.deg2rad(angle_azimuth)
    
    # Apply offset
    forward_offset = (
        forward_to_target * np.cos(angle_offset_rad) +
        (right * np.cos(angle_azimuth_rad) + up_local * np.sin(angle_azimuth_rad)) * np.sin(angle_offset_rad)
    )
    forward = normalize(forward_offset)
    
    # Apply pitch variation based on dynamic intensity
    # ALL modes add MORE variation than nadir (0°) or oblique (fixed angle)
    if mode == "mild_dynamic":
        # Mild dynamic: entry-level variation, still more than nadir/oblique
        pitch_adjustment = np.random.uniform(*MILD_DYNAMIC_PITCH_RANGE)
    elif mode == "moderate_dynamic":
        # Moderate dynamic: standard variation for robust testing
        pitch_adjustment = np.random.uniform(*MODERATE_DYNAMIC_PITCH_RANGE)
    else:  # extreme_dynamic
        # Extreme dynamic: maximum challenge for algorithm stress-testing
        pitch_adjustment = np.random.uniform(*EXTREME_DYNAMIC_PITCH_RANGE)
    
    # Add random roll (reduced)
    roll_deg = np.random.uniform(-10, 10)
    
    # Apply pitch adjustment
    pitch_rad = np.deg2rad(pitch_adjustment)
    forward_pitched = Rotation.from_rotvec(right * pitch_rad).apply(forward)
    
    # Apply roll
    roll_rad = np.deg2rad(roll_deg)
    forward_final = Rotation.from_rotvec(forward_pitched * roll_rad).apply(forward_pitched)
    right_final = Rotation.from_rotvec(forward_pitched * roll_rad).apply(right)
    
    # Calculate total pitch angle relative to local vertical
    vertical_angle = np.rad2deg(np.arccos(np.clip(np.dot(-forward_final, up), -1, 1)))
    pitch_deg = 90 - vertical_angle  # Convert to pitch (0°=horizontal, 90°=nadir)
    
    # Create quaternion
    quat = frame2quat(forward_final, right_final)
    
    return {
        "position": P_cam,
        "attitude": quat,
        "altitude": altitude,
        "alt_variation": alt_variation,
        "mode": mode,
        "pitch": pitch_deg,
        "roll": roll_deg,
        "baseline": baseline,
        "b_h_ratio": b_h_ratio,
        "azimuth": azimuth,
        "angle_offset_from_target": angle_offset
    }

# ════════════════════════════════════════════════════════════════════════════════
# MAIN SCRIPT
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer("127.0.0.1")
    s.closeViewer()
    s.setImageSize(512, 512)
    s.setCameraFOVDeg(45, 45)
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.enableRaytracing(True)

    s.setNbSamplesPerPixel(1)

    # Sun configuration
    pos_sun_init = vec3(0, 0, AU)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(3e16 * vec4(1,1,1,1))
    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "path/to/your/lunar.dem", "lambert", "")

    # Create spatial grid for uniform coverage
    print(f"Creating spatial grid for uniform coverage over zone {LAT_RANGE}")
    grid_points = create_spatial_grid(LAT_RANGE, LON_RANGE, N_LAT_BINS, N_LON_BINS)
    grid_points.sort(key=lambda p: abs(p[0] + 90))
    print(f"Number of grid points created: {len(grid_points)}")

    # Display generation plan
    print("\n" + "="*60)
    print("GENERATION PLAN BY ALTITUDE:")
    print("="*60)
    for alt_config in ALTITUDE_DISTRIBUTION:
        print(f"  Altitude {alt_config['altitude']:5d}m : {alt_config['pairs']:4d} pairs")
    print(f"  TOTAL             : {TOTAL_PAIRS:4d} pairs")
    print("="*60 + "\n")

    positions, attitudes, suns = [], [], []
    metadata = []
    
    # Global statistics
    total_valid_pairs = 0
    total_rejected_pairs = 0
    total_attempts = 0
    rejection_stats = {"infinity": 0, "invalid_depth": 0}
    
    # Global grid index
    grid_index = 0
    
    # Process each altitude level
    for alt_level, alt_config in enumerate(ALTITUDE_DISTRIBUTION):
        ALT_BASE = alt_config["altitude"]  # Reference altitude
        N_PAIRS = alt_config["pairs"]
        
        print(f"\n{'='*60}")
        print(f"Generating for reference altitude {ALT_BASE}m ({N_PAIRS} pairs)")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistics per altitude
        altitude_stats = {
            "modes": {"mild_dynamic": 0, "moderate_dynamic": 0, "extreme_dynamic": 0},
            "lat_coverage": {},
            "pitch_ranges": [],
            "roll_ranges": [],
            "baselines": [],
            "altitude_variations": []
        }

        while valid_pairs < N_PAIRS and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Use grid points first
            if grid_index < len(grid_points):
                lat, lon = grid_points[grid_index % len(grid_points)]
                grid_index += 1
            else:
                # If grid exhausted, generate randomly
                lon = np.random.uniform(*LON_RANGE)
                lat = np.random.uniform(*LAT_RANGE)

            # Terrain altitude (depth-map)
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, MOON_RADIUS, MOON_RADIUS)
            s.setObjectPosition("camera", vec3(*P_guess))
            s.setObjectAttitude("camera", look_at_quat(P_guess, [0,0,0]))
            s.render()
            
            # Get the full depth map
            depth_map = s.getDepthMap()

            # Filter finite values in (0, ALT_GUESS)
            mask = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < ALT_GUESS)
            valid_depths = depth_map[mask]

            if valid_depths.size == 0:
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue

            # Choose median depth
            depth = float(np.median(valid_depths))
            alt_ground = ALT_GUESS - depth

            # Surface point
            P_surf = geodetic_to_cartesian_BCBF_position(
                         lon, lat, alt_ground, MOON_RADIUS, MOON_RADIUS)
            
            # Define common target point (can be slightly offset from surface)
            # to ensure both cameras see a common zone
            target_offset = np.random.uniform(-500, 500)  # Target offset in meters
            up = normalize(P_surf)
            east = normalize(np.cross([0, 0, 1], up))
            if np.linalg.norm(east) < 1e-16:
                east = np.array([0, 1, 0])
            north = np.cross(up, east)
            
            # Target point with small random offset
            target_direction = np.random.uniform(0, 360)
            target_rad = np.deg2rad(target_direction)
            target_offset_vec = target_offset * (np.cos(target_rad) * north + np.sin(target_rad) * east)
            P_target = P_surf + target_offset_vec
            
            # Generate two camera configurations looking at the same zone
            # Use different constraint levels to vary configurations
            constraint_types = ["tight", "moderate", "loose"]
            constraint1 = np.random.choice(constraint_types, p=[0.3, 0.5, 0.2])
            constraint2 = np.random.choice(constraint_types, p=[0.3, 0.5, 0.2])
            
            cam1_config = generate_camera_config(P_surf, alt_ground, ALT_BASE, 1, P_target, constraint1)
            cam2_config = generate_camera_config(P_surf, alt_ground, ALT_BASE, 2, P_target, constraint2)
            
            # Check for infinity
            view1_ok, inf_count1 = check_infinity_in_view(s, cam1_config["position"], cam1_config["attitude"])
            view2_ok, inf_count2 = check_infinity_in_view(s, cam2_config["position"], cam2_config["attitude"])

            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Statistics
            altitude_stats["modes"][cam1_config["mode"]] += 1
            altitude_stats["modes"][cam2_config["mode"]] += 1
            altitude_stats["pitch_ranges"].extend([cam1_config["pitch"], cam2_config["pitch"]])
            altitude_stats["roll_ranges"].extend([cam1_config["roll"], cam2_config["roll"]])
            altitude_stats["baselines"].extend([cam1_config["baseline"], cam2_config["baseline"]])
            altitude_stats["altitude_variations"].extend([cam1_config["alt_variation"], cam2_config["alt_variation"]])
            
            # Calculate distance between cameras
            inter_camera_distance = np.linalg.norm(cam1_config["position"] - cam2_config["position"])
            
            # Display
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS} (total: {total_valid_pairs+valid_pairs+1}): "
                      f"Lat={lat:.2f}°, Lon={lon:.1f}°")
                print(f"  Camera 1: {cam1_config['mode']:15s} Alt={cam1_config['altitude']:.0f}m "
                      f"Pitch={cam1_config['pitch']:+.1f}° Offset={cam1_config['angle_offset_from_target']:.1f}° ({constraint1})")
                print(f"  Camera 2: {cam2_config['mode']:15s} Alt={cam2_config['altitude']:.0f}m "
                      f"Pitch={cam2_config['pitch']:+.1f}° Offset={cam2_config['angle_offset_from_target']:.1f}° ({constraint2})")
                print(f"  Inter-camera distance: {inter_camera_distance:.0f}m")

            # Record position
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1

            # Lighting configurations
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_surf)
                positions.extend([cam1_config["position"], cam2_config["position"]])
                attitudes.extend([cam1_config["attitude"], cam2_config["attitude"]])
                suns.extend([sun_pos, sun_pos])
                
                # Complete metadata for each image
                metadata.extend([
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 1, 
                        "mode": cam1_config["mode"],
                        "lat": lat, 
                        "lon": lon, 
                        "altitude_m": cam1_config["altitude"],
                        "altitude_variation": cam1_config["alt_variation"],
                        "pitch_deg": cam1_config["pitch"],
                        "roll_deg": cam1_config["roll"],
                        "baseline_m": cam1_config["baseline"],
                        "b_h_ratio": cam1_config["b_h_ratio"],
                        "baseline_azimuth": cam1_config["azimuth"],
                        "angle_offset_from_target": cam1_config["angle_offset_from_target"],
                        "constraint_type": constraint1,
                        "inter_camera_distance_m": inter_camera_distance,
                        "sun_azimuth": az,
                        "sun_incidence": inc
                    },
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 2, 
                        "mode": cam2_config["mode"],
                        "lat": lat, 
                        "lon": lon,
                        "altitude_m": cam2_config["altitude"],
                        "altitude_variation": cam2_config["alt_variation"],
                        "pitch_deg": cam2_config["pitch"],
                        "roll_deg": cam2_config["roll"],
                        "baseline_m": cam2_config["baseline"],
                        "b_h_ratio": cam2_config["b_h_ratio"],
                        "baseline_azimuth": cam2_config["azimuth"],
                        "angle_offset_from_target": cam2_config["angle_offset_from_target"],
                        "constraint_type": constraint2,
                        "inter_camera_distance_m": inter_camera_distance,
                        "sun_azimuth": az,
                        "sun_incidence": inc
                    }
                ])
            
            valid_pairs += 1
            
            if valid_pairs % 100 == 0:
                print(f"\nProgress altitude {ALT_BASE}m: {valid_pairs}/{N_PAIRS} pairs")

        # Summary for this altitude level
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Reference altitude {ALT_BASE}m completed:")
        print(f"  - Pairs generated: {valid_pairs}/{N_PAIRS}")
        print(f"  - Pairs rejected: {rejected_pairs}")
        print(f"  - Success rate: {100*valid_pairs/attempts:.1f}%")
        print(f"  - Mode distribution:")
        total_modes = sum(altitude_stats["modes"].values())
        for mode, count in altitude_stats["modes"].items():
            print(f"    • {mode}: {count} ({100*count/total_modes:.1f}%)")
        print(f"  - Mean pitch: {np.mean(altitude_stats['pitch_ranges']):.1f}° "
              f"(min: {np.min(altitude_stats['pitch_ranges']):.1f}°, "
              f"max: {np.max(altitude_stats['pitch_ranges']):.1f}°)")
        print(f"  - Mean baseline: {np.mean(altitude_stats['baselines']):.0f}m")

    # ════════════════════════════════════════════════════════════════════════════════
    # EXPORT FILES
    # ════════════════════════════════════════════════════════════════════════════════
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Camera trajectory
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("output/camera_trajectory_landing.csv", index=False)

    # Sun trajectory
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("output/sun_trajectory_landing.csv", index=False)
    
    # Detailed metadata
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("output/metadata_landing.csv", index=False)
    
    # Create summary per pair
    pair_summary = []
    for i in range(0, len(metadata), 6):  # 6 images per pair (2 cam × 3 lightings)
        cam1_data = metadata[i]
        cam2_data = metadata[i+1]
        pair_summary.append({
            "pair_id": cam1_data["pair_id"],
            "latitude": cam1_data["lat"],
            "longitude": cam1_data["lon"],
            "cam1_mode": cam1_data["mode"],
            "cam2_mode": cam2_data["mode"],
            "cam1_altitude_m": cam1_data["altitude_m"],
            "cam2_altitude_m": cam2_data["altitude_m"],
            "cam1_pitch_deg": cam1_data["pitch_deg"],
            "cam2_pitch_deg": cam2_data["pitch_deg"],
            "cam1_roll_deg": cam1_data["roll_deg"],
            "cam2_roll_deg": cam2_data["roll_deg"],
            "inter_camera_distance_m": cam1_data["inter_camera_distance_m"],
            "altitude_ref_m": ALT_BASE
        })
    
    df_summary = pd.DataFrame(pair_summary)
    df_summary.to_csv("output/pair_summary_landing.csv", index=False)

    # ════════════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FINAL GENERATION REPORT (VARIED STEREO)")
    print("="*80)
    
    print(f"\n Files exported:")
    print("   • output/camera_trajectory_landing.csv")
    print("   • output/sun_trajectory_landing.csv")
    print("   • output/metadata_landing.csv")
    print("   • output/pair_summary_landing.csv")
    
    print(f"\nGLOBAL STATISTICS:")
    print(f"  Total pairs generated: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)} ({total_valid_pairs} pairs × {N_LIGHTS} lightings × 2 cameras)")
    print(f"  Total rejections: {total_rejected_pairs}")
    print(f"    - Invalid depth: {rejection_stats['invalid_depth']}")
    print(f"    - Infinity pixels: {rejection_stats['infinity']}")
    print(f"  Global success rate: {100*total_valid_pairs/total_attempts:.1f}%")
    
    # Camera mode analysis
    print("\n" + "-"*60)
    print("GLOBAL CAMERA MODE DISTRIBUTION:")
    print("-"*60)
    mode_counts = df_meta.groupby('mode').size()
    total_cameras = len(df_meta)
    for mode in CAMERA_MODES:
        count = mode_counts.get(mode, 0)
        percentage = 100 * count / total_cameras
        print(f"  {mode:15s}: {count:5d} ({percentage:5.1f}%)")
    
    # Angle statistics
    print("\n" + "-"*60)
    print("ANGLE STATISTICS:")
    print("-"*60)
    print(f"  Pitch:")
    print(f"    - Mean: {df_meta['pitch_deg'].mean():.1f}°")
    print(f"    - Min/Max: {df_meta['pitch_deg'].min():.1f}° / {df_meta['pitch_deg'].max():.1f}°")
    print(f"  Roll:")
    print(f"    - Mean: {df_meta['roll_deg'].mean():.1f}°")
    print(f"    - Min/Max: {df_meta['roll_deg'].min():.1f}° / {df_meta['roll_deg'].max():.1f}°")
    
    # Inter-camera distance statistics
    print("\n" + "-"*60)
    print("INTER-CAMERA DISTANCES:")
    print("-"*60)
    print(f"  Mean: {df_summary['inter_camera_distance_m'].mean():.0f}m")
    print(f"  Min/Max: {df_summary['inter_camera_distance_m'].min():.0f}m / "
          f"{df_summary['inter_camera_distance_m'].max():.0f}m")
    
    # Mode combinations
    print("\n" + "-"*60)
    print("MOST FREQUENT MODE COMBINATIONS:")
    print("-"*60)
    mode_combos = df_summary.groupby(['cam1_mode', 'cam2_mode']).size().sort_values(ascending=False).head(10)
    for (mode1, mode2), count in mode_combos.items():
        percentage = 100 * count / len(df_summary)
        print(f"  {mode1:15s} + {mode2:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # Geographic distribution
    print("\n" + "-"*60)
    print("GEOGRAPHIC COVERAGE:")
    print("-"*60)
    lat_bins = pd.cut(df_summary['latitude'], bins=10)
    lat_distribution = df_summary.groupby(lat_bins).size()
    for interval, count in lat_distribution.items():
        percentage = 100 * count / len(df_summary)
        bar = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
        print(f"{interval}: {bar} {count:4d} ({percentage:5.1f}%)")
    
    # Summary by altitude
    print("\n" + "-"*60)
    print("SUMMARY BY REFERENCE ALTITUDE:")
    print("-"*60)
    print(f"{'Ref. Alt (m)':>12} | {'Target':>8} | {'Generated':>8} | {'Completed':>8}")
    print("-"*60)
    
    # Group by reference altitude
    altitude_counts = df_summary.groupby('altitude_ref_m').size()
    for alt_config in ALTITUDE_DISTRIBUTION:
        alt = alt_config["altitude"]
        target = alt_config["pairs"]
        actual = altitude_counts.get(alt, 0)
        completion = 100 * actual / target if target > 0 else 0
        print(f"{alt:>12} | {target:>8} | {actual:>8} | {completion:>7.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL':>12} | {TOTAL_PAIRS:>8} | {total_valid_pairs:>8} | {100*total_valid_pairs/TOTAL_PAIRS:>7.1f}%")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("Configuration: Stereo pairs with highly varied geometries")
    print("- Modes: mild_dynamic, moderate_dynamic, extreme_dynamic")
    print("- Altitude variations: ±30%")
    print("- Roll variations: ±10°")
    print(f"- B/H ratio: {B_H_RATIO_MIN*100:.0f}% to {B_H_RATIO_MAX*100:.0f}%")
    print("="*80)
