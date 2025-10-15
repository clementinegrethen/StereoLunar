import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4, quat, QuatToMat, gaussian, look_at
from PIL import Image
import math
import piexif  # pip install piexif
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from typing import Tuple
import json

# ────────────── CONFIGURATION GLOBALE ──────────────
moon_radius = 1_737_400

# NOUVEAU : Configuration de la distribution par altitude
TOTAL_PAIRS = 10  # Nombre total de paires à générer

# Définir les paliers d'altitude et leur répartition
ALTITUDE_DISTRIBUTION = [
    {"altitude": 24500,  "pairs": 1},   # 10%
    {"altitude": 26500,  "pairs": 1},   # 10%
    {"altitude": 28500,  "pairs": 1},   # 10%
    {"altitude": 30500, "pairs": 1},   # 10%
    {"altitude": 35500, "pairs": 1},   # 10%
    {"altitude": 40500, "pairs": 1},   # 10%
    {"altitude": 45500, "pairs": 1},   # 10%
    {"altitude": 50500, "pairs": 1},   # 10%
    {"altitude": 55550, "pairs": 1},   # 10%
    {"altitude": 60500, "pairs": 1},   # 10%
]

# NOUVEAU: Configuration du déplacement avec pitch - ADAPTÉE AU PETIT DEM
PITCH_ANGLE_RANGE = (5, 20)        # Angles plus petits pour rester dans le DEM
FORWARD_DISTANCE_RATIO = 0.05       # Déplacement très réduit (5% de l'altitude)
YAW_VARIATION_RANGE = (-10, 10)     # Variation de yaw très limitée

# Vérifier que le total correspond
total_check = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)
if total_check != TOTAL_PAIRS:
    print(f"⚠️ Attention: Total configuré ({total_check}) != TOTAL_PAIRS ({TOTAL_PAIRS})")
    # Ajuster automatiquement le dernier palier
    diff = TOTAL_PAIRS - total_check + ALTITUDE_DISTRIBUTION[-1]["pairs"]
    ALTITUDE_DISTRIBUTION[-1]["pairs"] = diff
    print(f"   Ajusté le dernier palier à {diff} paires")

# BASELINE : Ratio B/H aléatoire pour stéréo avec FOV 3°
B_H_RATIO_MIN = 0.002  # Ratio baseline/altitude minimum
B_H_RATIO_MAX = 0.015  # Réduit car FOV très petit (3°)

# Couverture du DEM LOLA - ZONE TRÈS RÉDUITE
LAT_RANGE   = (-46.08173, -46.18173)  # Seulement 0.1° de latitude
LON_RANGE   = (177.47, 177.52)        # Seulement 0.05° de longitude 

# Grille pour assurer une couverture uniforme
N_LAT_BINS = 25
N_LON_BINS = 50

# Paramètres pour filtrer l'infini
MAX_INFINITY_PIXELS = 0
DEPTH_CHECK_RESOLUTION = 128
INFINITY_THRESHOLD = 100_000_00

N_LIGHTS = 1
UA = 149_597_870_700

def generate_sun_setups():
    setups = []
    for k, az in enumerate(np.linspace(0, 360, N_LIGHTS, endpoint=False)):
        # UNIQUEMENT des angles très rasants pour des ombres profondes
        inc = np.random.uniform(120, 200)  # Lumière très rasante seulement
        setups.append((az, inc))
    return setups

SUN_SETUPS = generate_sun_setups()
print(f"Sun setups générés : {SUN_SETUPS}")

# ────────────── OUTILS ──────────────
def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-16 else v / n

def frame2quat(forward, right):
    z = normalize(forward)
    x = normalize(right)
    y = normalize(np.cross(z, x))
    R = np.column_stack((x, y, z))
    q = Rotation.from_matrix(R).as_quat()
    return (q[3], q[0], q[1], q[2])

def geodetic_to_cartesian_BCBF_position(lon, lat, alt, a, b):
    λ = np.deg2rad(((lon + 180) % 360) - 180)
    φ = np.deg2rad(lat)
    N = a**2 / np.sqrt(a**2*np.cos(φ)**2 + b**2*np.sin(φ)**2)
    X = (N+alt)*np.cos(φ)*np.cos(λ)
    Y = (N+alt)*np.cos(φ)*np.sin(λ)
    Z = ((b**2/a**2)*N+alt)*np.sin(φ)
    return np.array([X, Y, Z], dtype=np.float64)

def sun_position_local(az_deg, inc_deg, P_surf, dist=UA):
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

def create_pitched_attitude(position, pitch_deg, yaw_deg=0):
    """
    Crée une attitude avec pitch (angle par rapport au nadir) et yaw optionnel
    """
    up = normalize(position)
    east = normalize(np.cross([0, 0, 1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    north = np.cross(up, east)
    
    # Direction nadir (vers le bas)
    nadir = -up
    
    # Rotation autour de l'axe east pour le pitch
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)
    
    # Direction forward avec pitch
    forward = (np.cos(pitch_rad) * nadir + 
               np.sin(pitch_rad) * (np.cos(yaw_rad) * north + np.sin(yaw_rad) * east))
    forward = normalize(forward)
    
    # Right vector perpendiculaire
    right = normalize(np.cross(forward, up))
    
    return frame2quat(forward, right), forward

def generate_trajectory_pair(P_surf, altitude, baseline_ratio):
    """
    Génère une paire de positions/attitudes pour une trajectoire avec pitch
    ADAPTÉE POUR PETIT DEM - déplacements minimaux
    """
    up = normalize(P_surf)
    east = normalize(np.cross([0, 0, 1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    north = np.cross(up, east)
    
    # Position initiale (première caméra)
    P1 = P_surf + altitude * up
    
    # Angles pour la première caméra - TRÈS PETITS
    pitch1 = np.random.uniform(*PITCH_ANGLE_RANGE)
    yaw1 = np.random.uniform(*YAW_VARIATION_RANGE)
    
    quat1, forward1 = create_pitched_attitude(P1, pitch1, yaw1)
    
    # Déplacement MINIMAL pour la deuxième caméra
    # On privilégie le décalage latéral à l'avancement
    forward_ground = forward1 - np.dot(forward1, up) * up
    forward_ground = normalize(forward_ground)
    
    # Distances TRÈS RÉDUITES
    forward_distance = altitude * FORWARD_DISTANCE_RATIO  # Seulement 5%
    baseline_lateral = altitude * baseline_ratio
    
    # Position de la deuxième caméra - PRIORITÉ AU LATÉRAL
    lateral_direction = normalize(np.cross(up, forward_ground))
    
    # 80% latéral, 20% forward pour rester dans le DEM
    lateral_offset = np.random.choice([-1, 1]) * baseline_lateral * lateral_direction
    forward_offset = forward_distance * forward_ground * 0.2  # Très réduit
    
    P2_ground = P_surf + forward_offset + lateral_offset
    P2 = normalize(P2_ground) * (moon_radius + altitude)
    
    # Attitude de la deuxième caméra - VARIATION MINIMALE
    pitch2 = pitch1 + np.random.uniform(-2, 2)  # Variation très faible
    yaw2 = yaw1 + np.random.uniform(-3, 3)      # Variation très faible
    
    # S'assurer que les angles restent dans les limites
    pitch2 = np.clip(pitch2, PITCH_ANGLE_RANGE[0], PITCH_ANGLE_RANGE[1])
    yaw2 = np.clip(yaw2, YAW_VARIATION_RANGE[0], YAW_VARIATION_RANGE[1])
    
    quat2, forward2 = create_pitched_attitude(P2, pitch2, yaw2)
    
    return P1, P2, quat1, quat2, pitch1, pitch2, yaw1, yaw2

def check_infinity_in_view(s, position, attitude):
    """
    Vérifie si la vue contient des pixels à l'infini en utilisant la carte LOS.
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
        raise ValueError("La carte LOS n'a pas été générée correctement.")

    # Calculer la composante Z
    Z = depth_map * np.abs(los_map[..., 2])
    # Vérifier les pixels infinis dans la composante Z
    infinity_mask = (Z > INFINITY_THRESHOLD) | (~np.isfinite(Z))
    infinity_count = np.sum(infinity_mask)

    return infinity_count <= MAX_INFINITY_PIXELS, infinity_count

# Génération de grille spatiale
def create_spatial_grid(lat_range, lon_range, n_lat, n_lon):
    """
    Crée une grille de points uniformément répartis sur la surface
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

# ────────────── SCRIPT PRINCIPAL ──────────────
if __name__ == "__main__":
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer("127.0.0.1")
    s.closeViewer()
    s.setImageSize(512, 512)
    s.setCameraFOVDeg(3, 3)  # FOV très petit
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.enableRaytracing(True)
    s.setNbSamplesPerPixel(16)

    # Configuration du soleil
    pos_sun_init = vec3(0, 0, UA)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    pos_sun = vec3(150738070037.70956,7340266451.74839,-4123256661.86307)
    s.setObjectPosition("sun", pos_sun)
    s.setSunPower(3e16 * vec4(1,1,1,1))
    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "change4.dem", "lambert", "")
    s.enableLOSmapping(True)

    # Créer la grille de points
    print(f"Création d'une grille spatiale pour couvrir uniformément la zone {LAT_RANGE}")
    grid_points = create_spatial_grid(LAT_RANGE, LON_RANGE, N_LAT_BINS, N_LON_BINS)
    grid_points.sort(key=lambda p: abs(p[0] + 90))
    print(f"Nombre de points de grille créés: {len(grid_points)}")

    # Afficher le plan de génération
    print("\n" + "="*60)
    print("PLAN DE GÉNÉRATION AVEC TRAJECTOIRES PITCH:")
    print("="*60)
    print(f"CONFIGURATION ADAPTÉE POUR PETIT DEM:")
    print(f"Zone couverte: Lat {LAT_RANGE[0]:.3f}° à {LAT_RANGE[1]:.3f}°")
    print(f"               Lon {LON_RANGE[0]:.3f}° à {LON_RANGE[1]:.3f}°")
    print(f"Plage de pitch: {PITCH_ANGLE_RANGE[0]}° à {PITCH_ANGLE_RANGE[1]}° (réduite)")
    print(f"Distance de déplacement: {FORWARD_DISTANCE_RATIO*100:.1f}% de l'altitude (très réduite)")
    print(f"Variation de yaw: {YAW_VARIATION_RANGE[0]}° à {YAW_VARIATION_RANGE[1]}° (réduite)")
    print("="*60)
    for alt_config in ALTITUDE_DISTRIBUTION:
        print(f"  Altitude {alt_config['altitude']:5d}m : {alt_config['pairs']:4d} paires")
    print(f"  TOTAL             : {TOTAL_PAIRS:4d} paires")
    print("="*60 + "\n")

    positions, attitudes, suns = [], [], []
    metadata = []
    
    # Statistiques globales
    total_valid_pairs = 0
    total_rejected_pairs = 0
    total_attempts = 0
    rejection_stats = {"infinity": 0, "invalid_depth": 0}
    
    # Index global pour la grille
    grid_index = 0
    
    # Traiter chaque palier d'altitude
    for alt_level, alt_config in enumerate(ALTITUDE_DISTRIBUTION):
        ALT = alt_config["altitude"]
        N_PAIRS = alt_config["pairs"]
        
        print(f"\n{'='*60}")
        print(f"Génération trajectoires altitude {ALT}m ({N_PAIRS} paires)")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistiques par altitude
        altitude_stats = {
            "lat_coverage": {},
            "baseline_ratios": [],
            "sun_angles": [],
            "pitch_angles": [],
            "yaw_angles": []
        }

        while valid_pairs < N_PAIRS and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Position sur la grille - FIXÉE AU CENTRE DU PETIT DEM
            lat = -46.13173  # Centre de la petite zone
            lon = 177.495    # Centre de la petite zone
            
            # Ajouter une petite variation aléatoire pour éviter la répétition
            lat += np.random.uniform(-0.02, 0.02)  # ±0.02° de variation
            lon += np.random.uniform(-0.01, 0.01)  # ±0.01° de variation

            # Détermination de l'altitude du terrain
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, moon_radius, moon_radius)
            s.setObjectPosition("camera", vec3(*P_guess))
            
            # Attitude nadir temporaire pour mesurer le terrain
            up_guess = normalize(P_guess)
            nadir_forward = -up_guess
            east_guess = normalize(np.cross([0, 0, 1], up_guess))
            if np.linalg.norm(east_guess) < 1e-16:
                east_guess = np.array([0, 1, 0])
            nadir_quat = frame2quat(nadir_forward, east_guess)
            
            s.setObjectAttitude("camera", nadir_quat)
            s.render()
            
            # Échantillonnage de depth
            depth_samples = []
            for i in range(3):
                for j in range(3):
                    row = 256 + (i-1) * 50
                    col = 256 + (j-1) * 50
                    d = float(s.getDepthMap()[row, col])
                    depth_samples.append(d)
            
            if len(depth_samples) == 0:
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue
                
            depth = np.median(depth_samples)
            alt_ground = ALT_GUESS - depth
            
            # Position de surface
            P_surf = geodetic_to_cartesian_BCBF_position(
                         lon, lat, alt_ground, moon_radius, moon_radius)

            # Générer un ratio B/H adapté au FOV petit
            B_H_RATIO = np.random.uniform(B_H_RATIO_MIN, B_H_RATIO_MAX)
            altitude_stats["baseline_ratios"].append(B_H_RATIO)
            
            # Générer la trajectoire avec pitch
            P1, P2, qA, qB, pitch1, pitch2, yaw1, yaw2 = generate_trajectory_pair(
                P_surf, ALT, B_H_RATIO)
            
            # Enregistrer les angles pour les stats
            altitude_stats["pitch_angles"].extend([pitch1, pitch2])
            altitude_stats["yaw_angles"].extend([yaw1, yaw2])
            
            # Vérifier les vues
            view1_ok, inf_count1 = check_infinity_in_view(s, P1, qA)
            view2_ok, inf_count2 = check_infinity_in_view(s, P2, qB)
            
            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Calculs pour métadonnées
            baseline_distance = np.linalg.norm(P2 - P1)
            forward_distance = ALT * FORWARD_DISTANCE_RATIO * 0.2  # Distance réelle utilisée
            ground_coverage = 2 * ALT * np.tan(np.deg2rad(1.5))  # FOV/2 = 1.5°
            
            # Vérification que la couverture reste dans les limites du DEM
            dem_coverage_lat = abs(LAT_RANGE[1] - LAT_RANGE[0]) * 111_000  # ~11km par degré
            dem_coverage_lon = abs(LON_RANGE[1] - LON_RANGE[0]) * 111_000 * np.cos(np.deg2rad(lat))
            
            # Affichage avec info sur la couverture DEM
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS}: Lat={lat:.4f}°, Lon={lon:.4f}°")
                print(f"  Alt={ALT}m, Pitch={pitch1:.1f}°/{pitch2:.1f}°, Yaw={yaw1:.1f}°/{yaw2:.1f}°")
                print(f"  Baseline={baseline_distance:.0f}m, Forward={forward_distance:.0f}m")
                print(f"  Couverture sol={ground_coverage:.0f}m vs DEM={min(dem_coverage_lat, dem_coverage_lon):.0f}m")
            
            # Enregistrer la position
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1

            # Éclairages
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_surf)
                positions.extend([P1, P2])
                attitudes.extend([qA, qB])
                suns.extend([sun_pos, sun_pos])
                altitude_stats["sun_angles"].append(inc)
                
                # Metadata complète pour chaque image
                metadata.extend([
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 1, 
                        "lat": lat, 
                        "lon": lon, 
                        "altitude_m": ALT,
                        "baseline_m": baseline_distance, 
                        "b_h_ratio": B_H_RATIO,
                        "forward_distance_m": forward_distance,
                        "pitch_deg": pitch1,
                        "yaw_deg": yaw1,
                        "sun_azimuth": az,
                        "sun_incidence": inc,
                        "ground_coverage_m": ground_coverage
                    },
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 2, 
                        "lat": lat, 
                        "lon": lon,
                        "altitude_m": ALT,
                        "baseline_m": baseline_distance, 
                        "b_h_ratio": B_H_RATIO,
                        "forward_distance_m": forward_distance,
                        "pitch_deg": pitch2,
                        "yaw_deg": yaw2,
                        "sun_azimuth": az,
                        "sun_incidence": inc,
                        "ground_coverage_m": ground_coverage
                    }
                ])
            
            valid_pairs += 1

        # Résumé pour cette altitude
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude {ALT}m terminée:")
        print(f"  - Paires générées: {valid_pairs}/{N_PAIRS}")
        print(f"  - Pitch moyen: {np.mean(altitude_stats['pitch_angles']):.1f}°")
        print(f"  - Yaw moyen: {np.mean(altitude_stats['yaw_angles']):.1f}°")

    # Export des fichiers
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Trajectoire caméras
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("traj_real_pitch.csv", index=False)

    # Trajectoire Soleil
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("sun_real_pitch.csv", index=False)
    
    # Metadata détaillée
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("meta_pitch_trajectory.csv", index=False)
    
    # Créer un résumé par paire
    pair_summary = []
    for i in range(0, len(metadata), 2*N_LIGHTS):  # 2 cam × N_LIGHTS éclairages
        pair_data = metadata[i]
        pair_summary.append({
            "pair_id": pair_data["pair_id"],
            "latitude": pair_data["lat"],
            "longitude": pair_data["lon"],
            "altitude_m": pair_data["altitude_m"],
            "baseline_m": pair_data["baseline_m"],
            "forward_distance_m": pair_data["forward_distance_m"],
            "pitch_cam1_deg": pair_data["pitch_deg"],
            "pitch_cam2_deg": metadata[i+N_LIGHTS]["pitch_deg"],
            "yaw_cam1_deg": pair_data["yaw_deg"],
            "yaw_cam2_deg": metadata[i+N_LIGHTS]["yaw_deg"],
            "b_h_ratio": pair_data["b_h_ratio"],
            "ground_coverage_m": pair_data["ground_coverage_m"]
        })
    
    df_summary = pd.DataFrame(pair_summary)
    df_summary.to_csv("pair_summary_pitch_trajectory.csv", index=False)

    # ────────────── RAPPORT FINAL ──────────────
    print("\n" + "="*80)
    print("RAPPORT FINAL - TRAJECTOIRES AVEC PITCH")
    print("="*80)
    
    print(f"\n✅ Fichiers exportés:")
    print("   • traj_pitch_trajectory.csv")
    print("   • sun_pitch_trajectory.csv")
    print("   • meta_pitch_trajectory.csv")
    print("   • pair_summary_pitch_trajectory.csv")
    
    print(f"\nSTATISTIQUES GLOBALES:")
    print(f"  Total paires générées: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)}")
    print(f"  Total rejets: {total_rejected_pairs}")
    print(f"  Taux de succès global: {100*total_valid_pairs/total_attempts:.1f}%")
    
    # Statistiques des angles
    if len(df_summary) > 0:
        print(f"\nSTATISTIQUES DES ANGLES:")
        print(f"  Pitch moyen cam1: {df_summary['pitch_cam1_deg'].mean():.1f}° ±{df_summary['pitch_cam1_deg'].std():.1f}°")
        print(f"  Pitch moyen cam2: {df_summary['pitch_cam2_deg'].mean():.1f}° ±{df_summary['pitch_cam2_deg'].std():.1f}°")
        print(f"  Yaw moyen cam1: {df_summary['yaw_cam1_deg'].mean():.1f}° ±{df_summary['yaw_cam1_deg'].std():.1f}°")
        print(f"  Yaw moyen cam2: {df_summary['yaw_cam2_deg'].mean():.1f}° ±{df_summary['yaw_cam2_deg'].std():.1f}°")
        print(f"  Distance forward moyenne: {df_summary['forward_distance_m'].mean():.0f}m")
        print(f"  Baseline moyenne: {df_summary['baseline_m'].mean():.0f}m")
    
    print("\n" + "="*80)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
    print("Les caméras suivent maintenant des trajectoires avec pitch et déplacement!")
    print("="*80)