#!/usr/bin/env python3
"""
Oblique Stereo Image Pair Generator for Lunar Surface
======================================================
This script generates OBLIQUE (non-nadir) stereo image pairs of the lunar surface
using the SurRender rendering engine.

OBLIQUE VIEWING STRATEGY:
-------------------------
Unlike nadir imaging where cameras look straight down, oblique imaging positions
cameras at a horizontal distance from the target point, looking at it from an angle.
This creates viewing angles typically between 26° and 33° from horizontal.

Key differences from nadir:
- Cameras are positioned AWAY from the target point (not directly above)
- The viewing angle is computed as: angle = arctan(altitude / horizontal_distance)
- This reveals terrain features like crater walls, slopes, and shadows differently
- Better for 3D reconstruction of vertical structures

ALTITUDE VARIATION STRATEGIES:
------------------------------
To create diverse stereo pairs, four strategies vary camera altitudes:
1. "same": Both cameras at identical altitude (classical stereo)
2. "close": Small altitude difference (500-1000m) between cameras
3. "different": Large altitude difference (one high, one low)
4. "progressive": Sinusoidal variation based on pair index

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

# ────────────── GLOBAL CONFIGURATION ──────────────
MOON_RADIUS = 1_737_400  # Moon radius in meters

# Define altitude levels and their distribution
ALTITUDE_DISTRIBUTION = [
    {"altitude": 3500,  "pairs": 10},
    {"altitude": 6500,  "pairs": 10},
    {"altitude": 9500,  "pairs": 10},
    {"altitude": 12500, "pairs": 10},
    {"altitude": 15500, "pairs": 10},
    {"altitude": 18500, "pairs": 10},
    {"altitude": 21500, "pairs": 10},
    {"altitude": 24500, "pairs": 10},
    {"altitude": 27500, "pairs": 10},
    {"altitude": 30500, "pairs": 10},
]

# Total pairs is automatically computed from the altitude distribution
TOTAL_PAIRS = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)

# ════════════════════════════════════════════════════════════════════════════════
# OBLIQUE VIEW PARAMETERS
# ════════════════════════════════════════════════════════════════════════════════
# The key to oblique viewing is the distance-to-altitude ratio:
#   - Camera is placed at horizontal_distance = altitude × ratio
#   - View angle θ = arctan(altitude / distance)
#   - Ratio 0.5 → angle ≈ 63° from horizontal (27° from vertical)
#   - Ratio 1.1 → angle ≈ 42° from horizontal (48° from vertical)
# This keeps viewing angles consistent across different altitudes.

DISTANCE_TO_ALTITUDE_RATIO_MIN = 0.5  # Minimum ratio (steeper viewing angle ~63°)
DISTANCE_TO_ALTITUDE_RATIO_MAX = 1.1  # Maximum ratio (shallower viewing angle ~42°)
STEREO_BASELINE = 25  # Lateral separation between stereo cameras (meters)

# ════════════════════════════════════════════════════════════════════════════════
# ALTITUDE VARIATION STRATEGIES
# ════════════════════════════════════════════════════════════════════════════════
# These strategies control how camera altitudes vary within a stereo pair:
#   - "same": Classical stereo with identical camera heights
#   - "close": Small altitude difference for subtle parallax variation
#   - "different": Large altitude difference for extreme parallax
#   - "progressive": Smooth variation based on pair index (for dataset diversity)

ALTITUDE_STRATEGIES = ["same", "close", "different", "progressive"]
ALTITUDE_STRATEGY_WEIGHTS = [0.25, 0.25, 0.25, 0.25]  # Equal probability for each

# DEM coverage area (customize for your target region)
LAT_RANGE = (-89.9, -87.1)  # Latitude range (degrees)
LON_RANGE = (0.0, 360.0)    # Longitude range (degrees)

# Grid for uniform coverage
N_LAT_BINS = 25  # Number of latitude bands
N_LON_BINS = 50  # Number of longitude sectors

# Parameters for infinity filtering
MAX_INFINITY_PIXELS = 0             # Zero infinity pixels accepted
DEPTH_CHECK_RESOLUTION = 128        # Resolution for quick depth test
INFINITY_THRESHOLD = 100_000_00     # Beyond 100km, considered as infinity

# Lighting configuration
N_LIGHTS = 3
AU = 149_597_870_700  # Astronomical Unit in meters


def generate_sun_setups():
    """Generate sun position configurations for different lighting conditions."""
    setups = [
        (150, 160),  # Azimuth: 150°, Incidence: 160°
        (250, 20),   # Azimuth: 250°, Incidence: 20°
        (360, 165)   # Azimuth: 360°, Incidence: 165°
    ]
    return setups


SUN_SETUPS = generate_sun_setups()
print(f"Sun setups generated: {SUN_SETUPS}")

# ────────────── UTILITY FUNCTIONS ──────────────
def normalize(v):
    """Normalize a vector to unit length."""
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-16 else v / n


def frame2quat(forward, right):
    """Convert forward and right vectors to a quaternion (w, x, y, z)."""
    z = normalize(forward)
    x = normalize(right)
    y = normalize(np.cross(z, x))
    R = np.column_stack((x, y, z))
    q = Rotation.from_matrix(R).as_quat()
    return (q[3], q[0], q[1], q[2])


def geodetic_to_cartesian_BCBF_position(lon, lat, alt, a, b):
    """Convert geodetic coordinates (lon, lat, alt) to Cartesian BCBF position."""
    λ = np.deg2rad(((lon + 180) % 360) - 180)
    φ = np.deg2rad(lat)
    N = a**2 / np.sqrt(a**2*np.cos(φ)**2 + b**2*np.sin(φ)**2)
    X = (N+alt)*np.cos(φ)*np.cos(λ)
    Y = (N+alt)*np.cos(φ)*np.sin(λ)
    Z = ((b**2/a**2)*N+alt)*np.sin(φ)
    return np.array([X, Y, Z], dtype=np.float64)


def sun_position_local(az_deg, inc_deg, P_surf, dist=AU):
    """Compute sun position in local coordinates given azimuth and incidence angles."""
    az  = np.deg2rad(az_deg)
    inc = np.deg2rad(inc_deg)
    up    = normalize(P_surf)
    east  = normalize(np.cross([0, 0, 1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    north = np.cross(up, east)
    dir_local = (np.cos(inc)*(np.cos(az)*north + np.sin(az)*east)
                 + np.sin(inc)*up)
    return dir_local * dist


def create_spatial_grid(lat_range, lon_range, n_lat, n_lon):
    """
    Create a grid of uniformly distributed points on the surface,
    accounting for meridian convergence near the poles.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    grid_points = []
    
    for i in range(n_lat):
        lat_center = (lat_edges[i] + lat_edges[i+1]) / 2
        reduction_factor = max(0.3, abs(np.cos(np.deg2rad(lat_center))))
        n_lon_adjusted = max(8, int(n_lon * reduction_factor))
        lon_centers = np.linspace(lon_min, lon_max, n_lon_adjusted, endpoint=False)
        
        for lon in lon_centers:
            lat_jitter = np.random.uniform(-0.05, 0.05) * (lat_edges[i+1] - lat_edges[i])
            lon_jitter = np.random.uniform(-0.5, 0.5) * 360 / n_lon_adjusted
            final_lat = np.clip(lat_center + lat_jitter, lat_min, lat_max)
            final_lon = (lon + lon_jitter) % 360
            grid_points.append((final_lat, final_lon))
    
    return grid_points


# ════════════════════════════════════════════════════════════════════════════════
# OBLIQUE VIEW GENERATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def generate_altitude_pair(strategy, pair_index, alt_base, alt_range=2000):
    """
    Generate a pair of camera altitudes based on the selected strategy.
    
    This function implements different altitude pairing strategies to create
    diverse stereo configurations:
    
    Parameters:
    -----------
    strategy : str
        One of: "same", "close", "different", "progressive"
    pair_index : int
        Current pair index (used for progressive strategy)
    alt_base : float
        Base altitude around which to generate (meters)
    alt_range : float
        Range of altitude variation (meters)
    
    Returns:
    --------
    tuple : (alt1, alt2) - altitudes for camera 1 and camera 2
    
    Strategy Details:
    -----------------
    - "same": Both cameras at identical altitude
              → Classical stereo geometry, symmetric parallax
    
    - "close": Small altitude difference (500-1000m)
              → Slight asymmetry, useful for testing robustness
    
    - "different": One camera high, one low
              → Large parallax difference, challenges stereo matching
    
    - "progressive": Sinusoidal variation based on pair_index
              → Smooth coverage of altitude space across dataset
    """
    alt_min = max(1000, alt_base - alt_range//2)
    alt_max = alt_base + alt_range//2
    
    if strategy == "same":
        # Same altitude for both cameras - classical stereo configuration
        alt = np.random.uniform(alt_min, alt_max)
        return alt, alt
    
    elif strategy == "close":
        # Close altitudes (±500-1000m) - slight asymmetry
        alt1 = np.random.uniform(alt_min, alt_max)
        delta = np.random.uniform(500, 1000) * np.random.choice([-1, 1])
        alt2 = np.clip(alt1 + delta, alt_min, alt_max)
        return alt1, alt2
    
    elif strategy == "different":
        # One high, one low - maximum altitude diversity
        if np.random.random() < 0.5:
            alt1 = np.random.uniform(alt_min, alt_min + (alt_max - alt_min) * 0.3)
            alt2 = np.random.uniform(alt_max - (alt_max - alt_min) * 0.3, alt_max)
        else:
            alt1 = np.random.uniform(alt_max - (alt_max - alt_min) * 0.3, alt_max)
            alt2 = np.random.uniform(alt_min, alt_min + (alt_max - alt_min) * 0.3)
        return alt1, alt2
    
    elif strategy == "progressive":
        # Progressive variation based on pair index - sinusoidal pattern
        progress = pair_index / 100.0  # Normalize over 100 pairs
        osc = 0.5 * (1 + np.sin(2 * np.pi * 3 * progress))
        base_alt = alt_min + (alt_max - alt_min) * osc
        
        alt1 = base_alt + np.random.uniform(-500, 500)
        alt2 = base_alt + np.random.uniform(-500, 500)
        alt1 = np.clip(alt1, alt_min, alt_max)
        alt2 = np.clip(alt2, alt_min, alt_max)
        return alt1, alt2
    
    else:
        # Default: independent random altitudes
        return np.random.uniform(alt_min, alt_max), np.random.uniform(alt_min, alt_max)


def create_true_oblique_view(P_target, up_target, north, east, distance_horizontal, alt_cam, azimuth_deg):
    """
    Create an oblique camera view looking at the target point from a distance.
    
    OBLIQUE VIEW GEOMETRY:
    ----------------------
    Unlike nadir views where the camera is directly above the target,
    oblique views position the camera at a horizontal distance, creating
    an angled line of sight.
    
                    Camera (P_cam)
                      /
                     /  ← viewing angle θ
                    /
                   / distance_horizontal
                  /
                 /
    Target (P_target) ───────────────────
                      ground surface
    
    The viewing angle θ = arctan(altitude / distance_horizontal)
    
    Parameters:
    -----------
    P_target : array
        Target point position on the surface (Cartesian)
    up_target : array
        Local vertical direction at target
    north, east : array
        Local coordinate frame vectors at target
    distance_horizontal : float
        Horizontal distance from target to camera ground position (meters)
    alt_cam : float
        Camera altitude above local surface (meters)
    azimuth_deg : float
        Azimuth angle for viewing direction (degrees)
    
    Returns:
    --------
    tuple : (P_cam, quaternion, depression_angle)
        - P_cam: Camera position in Cartesian coordinates
        - quaternion: Camera orientation (w, x, y, z)
        - depression_angle: Angle below horizontal (radians)
    """
    # Compute view direction from azimuth
    az_rad = np.deg2rad(azimuth_deg)
    view_direction = -(np.cos(az_rad) * north + np.sin(az_rad) * east)
    
    # Position camera at horizontal distance from target
    # Camera is placed in the opposite direction of where it's looking
    P_cam_surface = P_target + distance_horizontal * view_direction
    
    # Normalize to sphere surface and add altitude
    P_cam_surface = normalize(P_cam_surface) * MOON_RADIUS
    up_cam = normalize(P_cam_surface)
    P_cam = P_cam_surface + alt_cam * up_cam
    
    # Forward vector: from camera toward target
    forward = normalize(P_target - P_cam)
    
    # Calculate depression angle (angle below horizontal)
    # Negative dot product because forward points toward target (down)
    angle_from_horizontal = np.arcsin(np.clip(-np.dot(forward, up_cam), -1, 1))
    
    # Right vector for camera frame
    right = normalize(np.cross(forward, up_cam))
    if np.linalg.norm(right) < 1e-16:
        right = east
    
    return P_cam, frame2quat(forward, right), angle_from_horizontal


def check_infinity_in_view(s, position, attitude):
    """
    Check if the view contains pixels at infinity using the LOS map.
    
    For oblique views, it's crucial to verify that the camera doesn't
    see beyond the horizon (infinity pixels), which would be invalid
    for depth-based applications.
    
    Returns:
    --------
    tuple : (is_valid, infinity_count)
        - is_valid: True if view has acceptable number of infinity pixels
        - infinity_count: Number of pixels seeing infinity
    """
    original_size = s.getImageSize()
    s.setImageSize(DEPTH_CHECK_RESOLUTION, DEPTH_CHECK_RESOLUTION)
    s.setObjectPosition("camera", vec3(*position))
    s.setObjectAttitude("camera", attitude)
    s.render()

    depth_map = s.getDepthMap()
    los_map = s.getLOSMap()

    s.setImageSize(original_size[0], original_size[1])

    if los_map is None:
        raise ValueError("LOS map was not generated correctly.")

    # Compute Z component (depth along viewing direction)
    Z = depth_map * np.abs(los_map[..., 2])
    
    # Check for infinity pixels
    infinity_mask = (Z > INFINITY_THRESHOLD) | (~np.isfinite(Z))
    infinity_count = np.sum(infinity_mask)

    return infinity_count <= MAX_INFINITY_PIXELS, infinity_count


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
    s.enableLOSmapping(True)
    s.setNbSamplesPerPixel(16)

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
    print("GENERATION PLAN BY ALTITUDE (WITH OBLIQUE VIEWS):")
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
    strategy_stats = {strategy: 0 for strategy in ALTITUDE_STRATEGIES}
    
    # Global grid index
    grid_index = 0
    
    # Process each altitude level
    for alt_level, alt_config in enumerate(ALTITUDE_DISTRIBUTION):
        ALT_BASE = alt_config["altitude"]
        N_PAIRS = alt_config["pairs"]
        
        print(f"\n{'='*60}")
        print(f"Generating for altitude {ALT_BASE}m ({N_PAIRS} pairs) - OBLIQUE VIEWS")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistics per altitude
        altitude_stats = {
            "lat_coverage": {},
            "strategies": {strategy: 0 for strategy in ALTITUDE_STRATEGIES},
            "oblique_angles": [],
            "distances": []
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
            s.setObjectAttitude("camera", frame2quat(normalize([0,0,0] - P_guess), [1,0,0]))
            s.render()
            
            # Get the full depth map
            depth_map = s.getDepthMap()
            mask = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < ALT_GUESS)
            valid_depths = depth_map[mask]

            if valid_depths.size == 0:
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue

            depth = float(np.median(valid_depths))
            alt_ground = ALT_GUESS - depth

            # Target point on the surface
            P_target = geodetic_to_cartesian_BCBF_position(lon, lat, alt_ground, MOON_RADIUS, MOON_RADIUS)
            up_target = normalize(P_target)
            
            # Calculate local base vectors
            east = normalize(np.cross([0,0,1], up_target))
            if np.linalg.norm(east) < 1e-16:
                east = np.array([0,1,0])
            north = np.cross(up_target, east)
            
            # Choose an altitude strategy
            strategy = np.random.choice(ALTITUDE_STRATEGIES, p=ALTITUDE_STRATEGY_WEIGHTS)
            alt1, alt2 = generate_altitude_pair(strategy, valid_pairs, ALT_BASE)
            
            # ─────────────────────────────────────────────────────────────
            # OBLIQUE VIEW: Distance parameters adaptive to altitude
            # The ratio between horizontal distance and altitude controls the
            # viewing angle: ratio = tan(θ) → θ = arctan(ratio)
            # With ratios 0.5-1.1, viewing angles range from ~26° to ~48°
            # ─────────────────────────────────────────────────────────────
            ratio1 = np.random.uniform(DISTANCE_TO_ALTITUDE_RATIO_MIN, DISTANCE_TO_ALTITUDE_RATIO_MAX)
            ratio2 = np.random.uniform(DISTANCE_TO_ALTITUDE_RATIO_MIN, DISTANCE_TO_ALTITUDE_RATIO_MAX)
            dist1 = alt1 * ratio1
            dist2 = alt2 * ratio2
            
            # Azimuth angles for stereo
            azimuth_base = np.random.uniform(0, 360)
            azimuth1 = azimuth_base + np.random.uniform(-10, 10)
            azimuth2 = azimuth_base + np.random.uniform(-10, 10)
            
            # Create the two oblique stereo views
            offset = STEREO_BASELINE * east * 0.5
            P1, q1, angle1 = create_true_oblique_view(P_target + offset, up_target, north, east, dist1, alt1, azimuth1)
            P2, q2, angle2 = create_true_oblique_view(P_target - offset, up_target, north, east, dist2, alt2, azimuth2)

            # Check for infinity pixels
            view1_ok, inf_count1 = check_infinity_in_view(s, P1, q1)
            view2_ok, inf_count2 = check_infinity_in_view(s, P2, q2)
            print(inf_count1)
            print(inf_count2)
            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Record statistics
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1
            altitude_stats["strategies"][strategy] += 1
            altitude_stats["oblique_angles"].extend([np.rad2deg(angle1), np.rad2deg(angle2)])
            altitude_stats["distances"].extend([dist1/1000, dist2/1000])
            strategy_stats[strategy] += 1

            # Periodic display
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS} (total: {total_valid_pairs+valid_pairs+1}): "
                      f"Lat={lat:.2f}°, Lon={lon:.1f}°")
                print(f"  Strategy={strategy}, Alt1={alt1:.0f}m, Alt2={alt2:.0f}m (ΔAlt={abs(alt2-alt1):.0f}m)")
                print(f"  Oblique angles: {np.rad2deg(angle1):.1f}°, {np.rad2deg(angle2):.1f}°")
                print(f"  Distances: {dist1/1000:.1f}km, {dist2/1000:.1f}km (ratios: {dist1/alt1:.1f}×, {dist2/alt2:.1f}×)")
                print(f"  Theoretical angles: {np.rad2deg(np.arctan(alt1/dist1)):.1f}°, {np.rad2deg(np.arctan(alt2/dist2)):.1f}°")

            # Record lighting configurations
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_target)
                positions.extend([P1, P2])
                attitudes.extend([q1, q2])
                suns.extend([sun_pos, sun_pos])
                
                # Complete metadata for each image
                metadata.extend([
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 1, 
                        "lat": lat, 
                        "lon": lon, 
                        "altitude_m": alt1,
                        "altitude_strategy": strategy,
                        "distance_km": dist1/1000,
                        "distance_ratio": dist1/alt1,
                        "theoretical_angle_deg": np.rad2deg(np.arctan(alt1/dist1)),
                        "oblique_angle_deg": np.rad2deg(angle1),
                        "sun_azimuth": az,
                        "sun_incidence": inc,
                        "target_altitude_m": ALT_BASE
                    },
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 2, 
                        "lat": lat, 
                        "lon": lon,
                        "altitude_m": alt2,
                        "altitude_strategy": strategy,
                        "distance_km": dist2/1000,
                        "distance_ratio": dist2/alt2,
                        "theoretical_angle_deg": np.rad2deg(np.arctan(alt2/dist2)),
                        "oblique_angle_deg": np.rad2deg(angle2),
                        "sun_azimuth": az,
                        "sun_incidence": inc,
                        "target_altitude_m": ALT_BASE
                    }
                ])
            
            valid_pairs += 1
            
            if valid_pairs % 100 == 0:
                print(f"\nProgress altitude {ALT_BASE}m: {valid_pairs}/{N_PAIRS} pairs")

        # Summary for this altitude level
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude {ALT_BASE}m completed:")
        print(f"  - Pairs generated: {valid_pairs}/{N_PAIRS}")
        print(f"  - Pairs rejected: {rejected_pairs}")
        print(f"  - Success rate: {100*valid_pairs/attempts:.1f}%")
        if altitude_stats["oblique_angles"]:
            print(f"  - Mean oblique angle: {np.mean(altitude_stats['oblique_angles']):.1f}°")
            print(f"  - Mean distance: {np.mean(altitude_stats['distances']):.1f}km")

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
    df_cam.to_csv("output/camera_trajectory_oblique.csv", index=False)

    # Sun trajectory
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("output/sun_trajectory_oblique.csv", index=False)
    
    # Detailed metadata
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("output/metadata_oblique.csv", index=False)
    
    # Create summary per pair
    pair_summary = []
    for i in range(0, len(metadata), 6):  # 6 images per pair (2 cam × 3 lightings)
        pair_data = metadata[i]
        pair_summary.append({
            "pair_id": pair_data["pair_id"],
            "latitude": pair_data["lat"],
            "longitude": pair_data["lon"],
            "altitude_strategy": pair_data["altitude_strategy"],
            "alt1_m": pair_data["altitude_m"],
            "alt2_m": metadata[i+1]["altitude_m"],  # Second camera
            "target_altitude_m": pair_data["target_altitude_m"],
            "distance1_km": pair_data["distance_km"],
            "distance2_km": metadata[i+1]["distance_km"],
            "distance_ratio1": pair_data["distance_ratio"],
            "distance_ratio2": metadata[i+1]["distance_ratio"],
            "oblique_angle1_deg": pair_data["oblique_angle_deg"],
            "oblique_angle2_deg": metadata[i+1]["oblique_angle_deg"]
        })
    
    df_summary = pd.DataFrame(pair_summary)
    df_summary.to_csv("output/pair_summary_oblique.csv", index=False)

    # ════════════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FINAL GENERATION REPORT - OBLIQUE VIEWS BY ALTITUDE")
    print("="*80)
    
    print(f"\nFiles exported:")
    print("   • output/camera_trajectory_oblique.csv")
    print("   • output/sun_trajectory_oblique.csv")
    print("   • output/metadata_oblique.csv")
    print("   • output/pair_summary_oblique.csv")
    
    print(f"\nGLOBAL STATISTICS:")
    print(f"  Total pairs generated: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)} ({total_valid_pairs} pairs × {N_LIGHTS} lightings × 2 cameras)")
    print(f"  Total rejections: {total_rejected_pairs}")
    print(f"    - Invalid depth: {rejection_stats['invalid_depth']}")
    print(f"    - Infinity pixels: {rejection_stats['infinity']}")
    print(f"  Global success rate: {100*total_valid_pairs/total_attempts:.1f}%")
    
    # Summary by altitude
    print("\n" + "-"*60)
    print("SUMMARY BY ALTITUDE:")
    print("-"*60)
    print(f"{'Altitude (m)':>12} | {'Target':>8} | {'Generated':>8} | {'Completed':>8}")
    print("-"*60)
    
    altitude_counts = df_summary.groupby('target_altitude_m').size()
    for alt_config in ALTITUDE_DISTRIBUTION:
        alt = alt_config["altitude"]
        target = alt_config["pairs"]
        actual = altitude_counts.get(alt, 0)
        completion = 100 * actual / target if target > 0 else 0
        print(f"{alt:>12} | {target:>8} | {actual:>8} | {completion:>7.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL':>12} | {TOTAL_PAIRS:>8} | {total_valid_pairs:>8} | {100*total_valid_pairs/TOTAL_PAIRS:>7.1f}%")
    
    # Strategy distribution
    print("\n" + "-"*60)
    print("ALTITUDE STRATEGY DISTRIBUTION:")
    print("-"*60)
    for strategy, count in strategy_stats.items():
        percentage = 100 * count / total_valid_pairs if total_valid_pairs > 0 else 0
        print(f"  {strategy:12s}: {count:4d} pairs ({percentage:5.1f}%)")
    
    # Oblique angle statistics
    print("\n" + "-"*60)
    print("OBLIQUE ANGLE STATISTICS:")
    print("-"*60)
    angles = df_meta['oblique_angle_deg']
    print(f"  Mean: {angles.mean():.1f}°")
    print(f"  Median: {angles.median():.1f}°")
    print(f"  Min: {angles.min():.1f}°, Max: {angles.max():.1f}°")
    print(f"  Standard deviation: {angles.std():.1f}°")
    
    # Distance and ratio statistics
    print(f"\n  Distances and ratios:")
    distances = df_meta['distance_km']
    ratios = df_meta['distance_ratio']
    print(f"  Mean distance: {distances.mean():.1f}km")
    print(f"  Min: {distances.min():.1f}km, Max: {distances.max():.1f}km")
    print(f"  Mean distance/altitude ratio: {ratios.mean():.1f}×")
    print(f"  Ratio min: {ratios.min():.1f}×, max: {ratios.max():.1f}×")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("Cameras use oblique views with distances adaptive to altitude")
    print(f"Distance/altitude ratios: {DISTANCE_TO_ALTITUDE_RATIO_MIN:.1f}× to {DISTANCE_TO_ALTITUDE_RATIO_MAX:.1f}×")
    print("Viewing angles maintained between ~26° and ~48° depending on altitude")
    print("="*80)
