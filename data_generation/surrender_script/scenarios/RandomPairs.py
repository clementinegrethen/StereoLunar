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
   # 10%
    {"altitude": 24500,  "pairs": 20},   # 10%
    {"altitude": 26500,  "pairs": 20},   # 10%
    {"altitude": 28500,  "pairs": 20},   # 10%
    {"altitude": 30500, "pairs": 20},   # 10%
    {"altitude": 35500, "pairs": 20},   # 10%
    {"altitude": 40500, "pairs": 20},   # 10%
    {"altitude": 45500, "pairs": 20},   # 10%
    {"altitude": 50500, "pairs": 20},   # 10%
    {"altitude": 55550, "pairs": 20},   # 10%
    {"altitude": 60500, "pairs": 20},   # 10%
  # 10%
]

# Vérifier que le total correspond
total_check = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)
if total_check != TOTAL_PAIRS:
    print(f"⚠️ Attention: Total configuré ({total_check}) != TOTAL_PAIRS ({TOTAL_PAIRS})")
    # Ajuster automatiquement le dernier palier
    diff = TOTAL_PAIRS - total_check + ALTITUDE_DISTRIBUTION[-1]["pairs"]
    ALTITUDE_DISTRIBUTION[-1]["pairs"] = diff
    print(f"   Ajusté le dernier palier à {diff} paires")

# BASELINE : Ratio B/H aléatoire pour stéréo avec FOV 30°
B_H_RATIO_MIN = 0.002  # Ratio baseline/altitude minimum
B_H_RATIO_MAX = 0.03   # Ratio baseline/altitude maximum
128, -89.70155682
# Couverture complète du DEM LOLA
LAT_RANGE   = (-46.08173, -47.08173)  # Couverture complète du pôle sud
LON_RANGE   = (177.47, 177.4) 

# Grille pour assurer une couverture uniforme
N_LAT_BINS = 25  # Nombre de bandes de latitude
N_LON_BINS = 50  # Nombre de secteurs de longitude

# Paramètres pour filtrer l'infini
MAX_INFINITY_PIXELS = 0             # ZÉRO pixel à l'infini accepté
DEPTH_CHECK_RESOLUTION = 128        # Résolution pour le test rapide de depth
INFINITY_THRESHOLD = 100_000_00     # Au-delà de 100km, on considère comme infini

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

def look_at_quat(eye_pos, target_pos):
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
def yawpitchroll_to_cartesian_BCBF_attitude(
    yaw_deg: float, pitch_deg: float, roll_deg: float, pos: Tuple[float, float, float]
) -> Tuple[float, float, float, float]:
    """
    Calcule l'attitude locale sous forme de quaternion (w, x, y, z)
    à partir de yaw, pitch, roll et d'une position (pour orienter "localement").
    """
    up = normalize(np.array(pos))
    east = np.cross([0, 0, 1], up)
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    else:
        east = normalize(east)
    north = np.cross(up, east)
    Rned = np.hstack((north[:, np.newaxis], east[:, np.newaxis], -up[:, np.newaxis]))
    Rypr = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    att = Rotation.from_matrix(Rned @ Rypr).as_quat()
    # Conversion du format [x,y,z,w] => (w,x,y,z)
    att = (att[3], *att[0:3])
    return att
# Génération de grille spatiale
def create_spatial_grid(lat_range, lon_range, n_lat, n_lon):
    """
    Crée une grille de points uniformément répartis sur la surface
    en tenant compte de la convergence des méridiens près du pôle
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

# Vérifier si une vue contient de l'infini
def check_infinity_in_view(s, position, attitude):
    """
    Vérifie si la vue contient des pixels à l'infini en utilisant la carte LOS.
    Retourne (True si OK, nombre de pixels infinis)
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


# ────────────── SCRIPT PRINCIPAL ──────────────
if __name__ == "__main__":
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer("127.0.0.1")
    s.closeViewer()
    s.setImageSize(512, 512)
    s.setCameraFOVDeg(3, 3)
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
    s.setSunPower(3e16 * vec4(1,1,1,1))  # Réduit pour des images plus sombres
    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "change4.dem", "lambert", "")
    s.enableLOSmapping(True)

    # Créer la grille de points pour une couverture uniforme
    print(f"Création d'une grille spatiale pour couvrir uniformément la zone {LAT_RANGE}")
    grid_points = create_spatial_grid(LAT_RANGE, LON_RANGE, N_LAT_BINS, N_LON_BINS)
    grid_points.sort(key=lambda p: abs(p[0] + 90))
    print(f"Nombre de points de grille créés: {len(grid_points)}")

    # Afficher le plan de génération
    print("\n" + "="*60)
    print("PLAN DE GÉNÉRATION PAR ALTITUDE:")
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
        print(f"Génération pour altitude {ALT}m ({N_PAIRS} paires)")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistiques par altitude
        altitude_stats = {
            "lat_coverage": {},
            "baseline_ratios": [],
            "sun_angles": []
        }

        while valid_pairs < N_PAIRS and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Utiliser les points de la grille en priorité
            if grid_index < len(grid_points):
                lat, lon = -46.08173,177.48291
                grid_index += 1
            else:
                # Si on a épuisé la grille, générer aléatoirement
                lon = 177.48291
                lat = -46.08173

            # altitude terrain (depth-map)
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, moon_radius, moon_radius)
            s.setObjectPosition("camera", vec3(*P_guess))
            nadir = yawpitchroll_to_cartesian_BCBF_attitude(0, 0, 0, P_guess)
            s.setObjectAttitude("camera", nadir)
            s.render()
            
            # Prendre plusieurs échantillons de depth
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
            print(alt_ground)
            # Caméras
            P_surf = geodetic_to_cartesian_BCBF_position(
                         lon, lat, alt_ground, moon_radius, moon_radius)
            up = normalize(P_surf)
            P1 = P_surf + ALT * up
            forward1 = normalize(P_surf - P1)
            right = normalize(np.cross(forward1, up))
            if np.linalg.norm(right) < 1e-16:
                right = np.array([0,1,0])
            qA = frame2quat(forward1, right)

            # Générer un ratio B/H aléatoire
            B_H_RATIO = np.random.uniform(B_H_RATIO_MIN, B_H_RATIO_MAX)
            altitude_stats["baseline_ratios"].append(B_H_RATIO)
            
            # Baseline adaptative
            BASE_ADAPTIVE = ALT * B_H_RATIO
            
            # Calculs pour info
            ground_coverage = 2 * ALT * np.tan(np.deg2rad(15))
            convergence_angle = np.rad2deg(np.arctan(BASE_ADAPTIVE / (2 * ALT)))
            
            # Affichage
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS} (total: {total_valid_pairs+valid_pairs+1}): "
                      f"Lat={lat:.2f}°, Lon={lon:.1f}°, Alt={ALT}m")
                print(f"  Baseline={BASE_ADAPTIVE:.0f}m (ratio B/H={B_H_RATIO:.3f})")
            
            P2_tan = P1 + np.random.choice([-1,1]) * BASE_ADAPTIVE * right
            P2 = normalize(P2_tan) * (moon_radius + alt_ground + ALT)
            forward2 = normalize(P_surf - P2)
            qB = frame2quat(forward2, right)

            view1_ok, inf_count1 = check_infinity_in_view(s, P1, qA)
            view2_ok, inf_count2 = check_infinity_in_view(s, P2, qB)
            
            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Enregistrer la position
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1

            # 4 éclairages
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

        # Résumé pour cette altitude
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude {ALT}m terminée:")
        print(f"  - Paires générées: {valid_pairs}/{N_PAIRS}")
        print(f"  - Paires rejetées: {rejected_pairs}")
        if attempts > 0:
            print(f"  - Taux de succès: {100*valid_pairs/attempts:.1f}%")
        else:
            print("  - Aucun essai effectué.")
        print(f"  - Ratio B/H moyen: {np.mean(altitude_stats['baseline_ratios']):.3f}")

    # Export des fichiers
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Trajectoire caméras
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("traj_val.csv", index=False)

    # Trajectoire Soleil
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("sun_val.csv", index=False)
    
    # Metadata détaillée
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("meta_val.csv", index=False)
    
    # Créer un résumé par paire (une ligne par paire au lieu de par image)
    pair_summary = []
    for i in range(0, len(metadata), 8):  # 8 images par paire (2 cam × 4 éclairages)
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
    df_summary.to_csv("pair_summary_altitude_distribution.csv", index=False)

    # ────────────── RAPPORT FINAL ──────────────
    print("\n" + "="*80)
    print("RAPPORT FINAL DE GÉNÉRATION")
    print("="*80)
    
    print(f"\n✅ Fichiers exportés:")
    print("   • traj_stereo_altitude_distribution.csv")
    print("   • sun_traj_altitude_distribution.csv")
    print("   • metadata_altitude_distribution.csv")
    print("   • pair_summary_altitude_distribution.csv")
    
    print(f"\nSTATISTIQUES GLOBALES:")
    print(f"  Total paires générées: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)} ({total_valid_pairs} paires × {N_LIGHTS} éclairages × 2 caméras)")
    print(f"  Total rejets: {total_rejected_pairs}")
    print(f"    - Depth invalide: {rejection_stats['invalid_depth']}")
    print(f"    - Pixels infinis: {rejection_stats['infinity']}")
    print(f"  Taux de succès global: {100*total_valid_pairs/total_attempts:.1f}%")
    
    # Récapitulatif par altitude
    print("\n" + "-"*60)
    print("RÉCAPITULATIF PAR ALTITUDE:")
    print("-"*60)
    print(f"{'Altitude (m)':>12} | {'Objectif':>8} | {'Générées':>8} | {'Complété':>8}")
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
    
    # Distribution géographique
    print("\n" + "-"*60)
    print("COUVERTURE GÉOGRAPHIQUE:")
    print("-"*60)
    lat_bins = pd.cut(df_summary['latitude'], bins=10)
    lat_distribution = df_summary.groupby(lat_bins).size()
    for interval, count in lat_distribution.items():
        percentage = 100 * count / len(df_summary)
        bar = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
        print(f"{interval}: {bar} {count:4d} ({percentage:5.1f}%)")
    
    # Statistiques des baselines
    print("\n" + "-"*60)
    print("STATISTIQUES DES BASELINES:")
    print("-"*60)
    for alt in sorted(altitude_counts.index):
        alt_data = df_summary[df_summary['altitude_m'] == alt]
        print(f"\nAltitude {alt}m:")
        print(f"  Baseline moyenne: {alt_data['baseline_m'].mean():.1f}m")
        print(f"  Ratio B/H moyen: {alt_data['b_h_ratio'].mean():.3f}")
        print(f"  Couverture sol moyenne: {alt_data['ground_coverage_m'].mean():.0f}m")
        print(f"  Angle convergence moyen: {alt_data['convergence_angle_deg'].mean():.2f}°")
    
    print("\n" + "="*80)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
    print("="*80)