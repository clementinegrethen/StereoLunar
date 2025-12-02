#!/usr/bin/env python3
"""
Nadir Stereo Image Pair Generator for Lunar Surface
====================================================
This script generates nadir-viewing stereo image pairs of the lunar surface
using the SurRender rendering engine. It creates pairs at various altitudes
with configurable baseline ratios for stereo reconstruction applications.


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


# here are the altitude for our configuration
# 10 altitude levels, each with an equal share of pairs
# In order to respect the gsd of the dem with a fov of 45
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

# BASELINE: Random B/H ratio for stereo with 45 FOV
#We used this configuration in the paper
# Feel free to adjust these values!B_H_RATIO_MIN = 0.02  # Minimum baseline/altitude ratio
B_H_RATIO_MAX = 0.1   # Maximum baseline/altitude ratio

# DEM coverage area 
# Latitude and Longitude ranges (in degrees)
# we set those values as an example, feel free to adjust them
LAT_RANGE = (-89.0, -87.1)  # Latitude range (degrees)
LON_RANGE = (0.0, 90)       # Longitude range (degrees)

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

def look_at_quat(eye_pos, target_pos):
    """Compute quaternion to orient camera from eye position looking at target."""
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
# to be sure there is no infinity in the view as the lola dem has some holes

def check_infinity_in_view(s, position, attitude):
    """
    Check if the view contains pixels at infinity.
    Returns (True if OK, number of infinity pixels).
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

# ────────────── MAIN SCRIPT ──────────────
if __name__ == "__main__":
    # Initialize SurRender client
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer("127.0.0.1")  # SurRender server address
    s.closeViewer()
    s.setImageSize(512, 512)
    s.setCameraFOVDeg(45, 45)
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.enableRaytracing(True)
    s.setNbSamplesPerPixel(16)

    # Sun configuration
    pos_sun_init = vec3(0, 0, AU)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(3e16 * vec4(1,1,1,1))
    s.createBRDF("hapke", "hapke.brdf", 0.12)
    # Load lunar DEM - replace with your DEM file path
    s.createSphericalDEM("moon_dem", "path/to/your/lunar.dem", "hapke", "")

    # Create grid points for uniform coverage
    print(f"Creating spatial grid to uniformly cover the area {LAT_RANGE}")
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
    
    # Global index for grid
    grid_index = 0
    
    # Process each altitude level
    for alt_level, alt_config in enumerate(ALTITUDE_DISTRIBUTION):
        ALT = alt_config["altitude"]
        N_PAIRS = alt_config["pairs"]
        
        print(f"\n{'='*60}")
        print(f"Generating for altitude {ALT}m ({N_PAIRS} pairs)")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistics per altitude
        altitude_stats = {
            "lat_coverage": {},
            "baseline_ratios": [],
            "sun_angles": []
        }

        while valid_pairs < N_PAIRS and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Use grid points first
            if grid_index < len(grid_points):
                lat, lon = grid_points[grid_index % len(grid_points)]
                grid_index += 1
            else:
                # If grid is exhausted, generate randomly
                lon = np.random.uniform(*LON_RANGE)
                lat = np.random.uniform(*LAT_RANGE)

            # Terrain altitude (depth-map)
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, MOON_RADIUS, MOON_RADIUS)
            s.setObjectPosition("camera", vec3(*P_guess))
            s.setObjectAttitude("camera", look_at_quat(P_guess, [0,0,0]))
            s.render()
            
           # Récupérer la depth map entière
            depth_map = s.getDepthMap()  # taille DEPTH_CHECK_RESOLUTION²

            # Filtrer les valeurs finies ET dans (0, ALT_GUESS)
            mask = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < ALT_GUESS)
            valid_depths = depth_map[mask]

            if valid_depths.size == 0:
                # Aucune profondeur valide → rejetez la paire
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue

            # Choisir la médiane (ou le percentile bas pour éviter les « trous »)
            depth = float(np.median(valid_depths))
            alt_ground = ALT_GUESS - depth

            print(f"grouuund: {alt_ground}")
            # Caméras
            P_surf = geodetic_to_cartesian_BCBF_position(
                         lon, lat, alt_ground, moon_radius, moon_radius)
            print(f"P_surf: {P_surf}")
            up = normalize(P_surf)
            P1 = P_surf + ALT * up
            print(f"P1: {P1}")
            forward1 = normalize(P_surf - P1)
            right = normalize(np.cross(forward1, up))
            if np.linalg.norm(right) < 1e-16:
                right = np.array([0,1,0])
            qA = frame2quat(forward1, right)

            # Generate random B/H ratio
            B_H_RATIO = np.random.uniform(B_H_RATIO_MIN, B_H_RATIO_MAX)
            altitude_stats["baseline_ratios"].append(B_H_RATIO)
            
            # Adaptive baseline
            BASE_ADAPTIVE = ALT * B_H_RATIO
            
            # Calculations for info
            ground_coverage = 2 * ALT * np.tan(np.deg2rad(15))
            convergence_angle = np.rad2deg(np.arctan(BASE_ADAPTIVE / (2 * ALT)))
            
            # Display progress
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS} (total: {total_valid_pairs+valid_pairs+1}): "
                      f"Lat={lat:.2f}°, Lon={lon:.1f}°, Alt={ALT}m")
                print(f"  Baseline={BASE_ADAPTIVE:.0f}m (ratio B/H={B_H_RATIO:.3f})")
            
            P2_tan = P1 + np.random.choice([-1,1]) * BASE_ADAPTIVE * right
            P2 = normalize(P2_tan) * (MOON_RADIUS + alt_ground + ALT)
            forward2 = normalize(P_surf - P2)
            qB = frame2quat(forward2, right)

            # Check for infinity
            view1_ok, inf_count1 = check_infinity_in_view(s, P1, qA)
            print(inf_count1)
            view2_ok, inf_count2 = check_infinity_in_view(s, P2, qB)
            print(inf_count2)

            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Record position
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1

            # Multiple lighting conditions
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_surf)
                positions.extend([P1, P2])
                attitudes.extend([qA, qB])
                suns.extend([sun_pos, sun_pos])
                altitude_stats["sun_angles"].append(inc)
                
                # Complete metadata for each image
                metadata.extend([
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 1, 
                        "lat": lat, 
                        "lon": lon, 
                        "altitude_m": ALT,
                        "baseline_m": BASE_ADAPTIVE, 
                        "b_h_ratio": B_H_RATIO,
                        "sun_azimuth": az,
                        "sun_incidence": inc,
                        "ground_coverage_m": ground_coverage,
                        "convergence_angle_deg": convergence_angle
                    },
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 2, 
                        "lat": lat, 
                        "lon": lon,
                        "altitude_m": ALT,
                        "baseline_m": BASE_ADAPTIVE, 
                        "b_h_ratio": B_H_RATIO,
                        "sun_azimuth": az,
                        "sun_incidence": inc,
                        "ground_coverage_m": ground_coverage,
                        "convergence_angle_deg": convergence_angle
                    }
                ])
            
            valid_pairs += 1
            
            if valid_pairs % 100 == 0:
                print(f"\nProgress altitude {ALT}m: {valid_pairs}/{N_PAIRS} pairs")

        # Summary for this altitude
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude {ALT}m completed:")
        print(f"  - Pairs generated: {valid_pairs}/{N_PAIRS}")
        print(f"  - Pairs rejected: {rejected_pairs}")
        print(f"  - Success rate: {100*valid_pairs/attempts:.1f}%")
        print(f"  - Average B/H ratio: {np.mean(altitude_stats['baseline_ratios']):.3f}")

    # Export files
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Camera trajectory
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("output/camera_trajectory.csv", index=False)

    # Sun trajectory
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("output/sun_trajectory.csv", index=False)
    
    # Detailed metadata
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("output/metadata.csv", index=False)
    
    # Create summary per pair (one row per pair instead of per image)
    pair_summary = []
    for i in range(0, len(metadata), 8):  # 8 images per pair (2 cam × N_LIGHTS)
        pair_data = metadata[i]
        pair_summary.append({
            "pair_id": pair_data["pair_id"],
            "latitude": pair_data["lat"],
            "longitude": pair_data["lon"],
            "altitude_m": pair_data["altitude_m"],
            "baseline_m": pair_data["baseline_m"],
            "b_h_ratio": pair_data["b_h_ratio"],
            "ground_coverage_m": pair_data["ground_coverage_m"],
            "convergence_angle_deg": pair_data["convergence_angle_deg"]
        })
    
    df_summary = pd.DataFrame(pair_summary)
    df_summary.to_csv("output/pair_summary.csv", index=False)

    # ────────────── FINAL REPORT ──────────────
    print("\n" + "="*80)
    print("FINAL GENERATION REPORT")
    print("="*80)
    
    print(f"\n✅ Files exported:")
    print("   • output/camera_trajectory.csv")
    print("   • output/sun_trajectory.csv")
    print("   • output/metadata.csv")
    print("   • output/pair_summary.csv")
    
    print(f"\nGLOBAL STATISTICS:")
    print(f"  Total pairs generated: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)} ({total_valid_pairs} pairs × {N_LIGHTS} lighting × 2 cameras)")
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
    
    altitude_counts = df_summary.groupby('altitude_m').size()
    for alt_config in ALTITUDE_DISTRIBUTION:
        alt = alt_config["altitude"]
        target = alt_config["pairs"]
        actual = altitude_counts.get(alt, 0)
        completion = 100 * actual / target if target > 0 else 0
        print(f"{alt:>12} | {target:>8} | {actual:>8} | {completion:>7.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL':>12} | {TOTAL_PAIRS:>8} | {total_valid_pairs:>8} | {100*total_valid_pairs/TOTAL_PAIRS:>7.1f}%")
    
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
    
    # Baseline statistics
    print("\n" + "-"*60)
    print("BASELINE STATISTICS:")
    print("-"*60)
    for alt in sorted(altitude_counts.index):
        alt_data = df_summary[df_summary['altitude_m'] == alt]
        print(f"\nAltitude {alt}m:")
        print(f"  Average baseline: {alt_data['baseline_m'].mean():.1f}m")
        print(f"  Average B/H ratio: {alt_data['b_h_ratio'].mean():.3f}")
        print(f"  Average ground coverage: {alt_data['ground_coverage_m'].mean():.0f}m")
        print(f"  Average convergence angle: {alt_data['convergence_angle_deg'].mean():.2f}°")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("="*80)
