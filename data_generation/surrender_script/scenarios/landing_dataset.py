import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4

# ────────────── CONFIGURATION GLOBALE ──────────────
moon_radius = 1_737_400

# NOUVEAU : Configuration de la distribution par altitude
TOTAL_PAIRS = 200  # Nombre total de paires à générer

# Définir les paliers d'altitude et leur répartition
ALTITUDE_DISTRIBUTION = [
    {"altitude": 3500,  "pairs": 20},   # 10%
    {"altitude": 6500,  "pairs": 20},   # 10%
    {"altitude": 9500,  "pairs": 20},   # 10%
    {"altitude": 12500, "pairs": 20},   # 10%
    {"altitude": 15500, "pairs": 20},   # 10%
    {"altitude": 18500, "pairs": 20},   # 10%
    {"altitude": 21500, "pairs": 20},   # 10%
    {"altitude": 24500, "pairs": 20},   # 10%
    {"altitude": 27500, "pairs": 20},   # 10%
    {"altitude": 30500, "pairs": 20},   # 10%
]

# Vérifier que le total correspond
total_check = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)
if total_check != TOTAL_PAIRS:
    print(f"⚠️ Attention: Total configuré ({total_check}) != TOTAL_PAIRS ({TOTAL_PAIRS})")
    # Ajuster automatiquement le dernier palier
    diff = TOTAL_PAIRS - total_check + ALTITUDE_DISTRIBUTION[-1]["pairs"]
    ALTITUDE_DISTRIBUTION[-1]["pairs"] = diff
    print(f"   Ajusté le dernier palier à {diff} paires")

# NOUVELLE CONFIGURATION STEREO PLUS VARIÉE
# Les deux caméras peuvent avoir des configurations très différentes
CAMERA_MODES = ["nadir", "oblique", "strong_oblique"]  # Modes de prise de vue

# Paramètres pour chaque mode
NADIR_PITCH_RANGE = (-5, 5)           # Presque vertical
OBLIQUE_PITCH_RANGE = (10, 20)        # Oblique modéré
STRONG_OBLIQUE_PITCH_RANGE = (20, 35) # Oblique fort

# Variation d'altitude pour chaque caméra (indépendamment)
ALT_VARIATION_RANGE = (-0.3, 0.3)  # +/- 30% de variation possible

# Variation de roll possible
ROLL_VARIATION_RANGE = (-10, 10)  # +/- 15° de roll

# BASELINE : Ratio B/H pour maintenir des angles cohérents
B_H_RATIO_MIN = 0.05  # Ratio baseline/altitude minimum (5%)
B_H_RATIO_MAX = 0.18  # Ratio baseline/altitude maximum (25%)

# Couverture complète du DEM LOLA
LAT_RANGE   = (-89.0, -87.1)  # Couverture complète du pôle sud
LON_RANGE   = (0.0, 360)

# Grille pour assurer une couverture uniforme
N_LAT_BINS = 25  # Nombre de bandes de latitude
N_LON_BINS = 50  # Nombre de secteurs de longitude

# Paramètres pour filtrer l'infini
MAX_INFINITY_PIXELS = 0             # ZÉRO pixel à l'infini accepté
DEPTH_CHECK_RESOLUTION = 128        # Résolution pour le test rapide de depth
INFINITY_THRESHOLD = 100_000_00     # Au-delà de 100km, on considère comme infini

N_LIGHTS = 3
UA = 149_597_870_700

def generate_sun_setups():
    setups = [
        (150, 160),  # Azimuth: 45°, Incidence: 135°
        (250, 20), # Azimuth: 135°, Incidence: 150°
        (360, 165)  # Azimuth: 225°, Incidence: 165°
    ]
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
    Vérifie si la vue contient des pixels à l'infini
    Retourne (True si OK, nombre de pixels infinis)
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
    Génère une configuration aléatoire pour une caméra avec contrainte de visée
    
    Args:
        P_surf: Point de surface de référence
        alt_ground: Altitude du terrain
        base_altitude: Altitude de base du palier
        camera_id: Identifiant de la caméra (1 ou 2)
        target_point: Point cible à voir (si None, utilise P_surf)
        constraint_type: Type de contrainte ("tight", "moderate", "loose")
    """
    if target_point is None:
        target_point = P_surf
    
    # Choisir un mode aléatoire
    mode = np.random.choice(CAMERA_MODES, p=[0.3, 0.4, 0.3])  # 30% nadir, 40% oblique, 30% strong oblique
    
    # Variation d'altitude
    alt_variation = np.random.uniform(*ALT_VARIATION_RANGE)
    altitude = base_altitude * (1 + alt_variation)
    
    # Position de base
    up = normalize(P_surf)
    
    # Décalage latéral aléatoire pour la position
    east = normalize(np.cross([0, 0, 1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    north = np.cross(up, east)
    
    # BASELINE PROPORTIONNELLE À L'ALTITUDE
    # Générer un ratio B/H aléatoire
    b_h_ratio = np.random.uniform(B_H_RATIO_MIN, B_H_RATIO_MAX)
    baseline = altitude * b_h_ratio
    
    # Direction aléatoire pour la baseline
    azimuth = np.random.uniform(0, 360)
    az_rad = np.deg2rad(azimuth)
    
    lateral_offset = baseline * (np.cos(az_rad) * north + np.sin(az_rad) * east)
    
    # Position de la caméra
    P_cam_nominal = P_surf + altitude * up + lateral_offset
    P_cam = normalize(P_cam_nominal) * (moon_radius + alt_ground + altitude)
    
    # Direction de base vers le point cible (garantit la zone commune)
    forward_to_target = normalize(target_point - P_cam)
    
    # Créer un système de coordonnées local
    right = normalize(np.cross(forward_to_target, up))
    if np.linalg.norm(right) < 1e-16:
        right = normalize(np.cross(forward_to_target, north))
    up_local = np.cross(right, forward_to_target)
    
    # Appliquer un décalage angulaire limité selon le type de contrainte
    if constraint_type == "tight":
        max_angle_offset = 10  # Maximum 10° de décalage par rapport au point cible
    elif constraint_type == "moderate":
        max_angle_offset = 20  # Maximum 20° de décalage
    else:  # loose
        max_angle_offset = 30  # Maximum 30° de décalage
    
    # Décalage angulaire aléatoire dans le cône autour de la direction cible
    angle_offset = np.random.uniform(0, max_angle_offset)
    angle_azimuth = np.random.uniform(0, 360)
    
    # Convertir en radians
    angle_offset_rad = np.deg2rad(angle_offset)
    angle_azimuth_rad = np.deg2rad(angle_azimuth)
    
    # Appliquer le décalage
    forward_offset = (
        forward_to_target * np.cos(angle_offset_rad) +
        (right * np.cos(angle_azimuth_rad) + up_local * np.sin(angle_azimuth_rad)) * np.sin(angle_offset_rad)
    )
    forward = normalize(forward_offset)
    
    # Maintenant appliquer les rotations supplémentaires selon le mode
    # Mais avec des valeurs réduites pour ne pas perdre la zone commune
    if mode == "nadir":
        pitch_adjustment = np.random.uniform(-5, 5)
    elif mode == "oblique":
        pitch_adjustment = np.random.uniform(-10, 10)
    else:  # strong_oblique
        pitch_adjustment = np.random.uniform(-15, 15)
    
    # Ajouter du roll aléatoire (réduit)
    roll_deg = np.random.uniform(-10, 10)
    
    # Appliquer pitch adjustment
    pitch_rad = np.deg2rad(pitch_adjustment)
    forward_pitched = Rotation.from_rotvec(right * pitch_rad).apply(forward)
    
    # Appliquer roll
    roll_rad = np.deg2rad(roll_deg)
    forward_final = Rotation.from_rotvec(forward_pitched * roll_rad).apply(forward_pitched)
    right_final = Rotation.from_rotvec(forward_pitched * roll_rad).apply(right)
    
    # Calculer l'angle de pitch total par rapport à la verticale locale
    vertical_angle = np.rad2deg(np.arccos(np.clip(np.dot(-forward_final, up), -1, 1)))
    pitch_deg = 90 - vertical_angle  # Convertir en pitch (0° = horizontal, 90° = nadir)
    
    # Créer le quaternion
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

# ────────────── SCRIPT PRINCIPAL ──────────────
if __name__ == "__main__":
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer("127.0.0.1")
    s.closeViewer()
    s.setImageSize(512, 512)
    s.setCameraFOVDeg(30, 30)
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.enableRaytracing(True)
    s.setNbSamplesPerPixel(16)

    # Configuration du soleil
    pos_sun_init = vec3(0, 0, UA)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(3e16 * vec4(1,1,1,1))  # Réduit pour des images plus sombres
    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "south5m.dem", "lambert", "")

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
        ALT_BASE = alt_config["altitude"]  # Altitude de référence
        N_PAIRS = alt_config["pairs"]
        
        print(f"\n{'='*60}")
        print(f"Génération pour altitude de référence {ALT_BASE}m ({N_PAIRS} paires)")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistiques par altitude
        altitude_stats = {
            "modes": {"nadir": 0, "oblique": 0, "strong_oblique": 0},
            "lat_coverage": {},
            "pitch_ranges": [],
            "roll_ranges": [],
            "baselines": [],
            "altitude_variations": []
        }

        while valid_pairs < N_PAIRS and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Utiliser les points de la grille en priorité
            if grid_index < len(grid_points):
                lat, lon = grid_points[grid_index % len(grid_points)]
                grid_index += 1
            else:
                # Si on a épuisé la grille, générer aléatoirement
                lon = np.random.uniform(*LON_RANGE)
                lat = np.random.uniform(*LAT_RANGE)

            # altitude terrain (depth-map)
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, moon_radius, moon_radius)
            s.setObjectPosition("camera", vec3(*P_guess))
            s.setObjectAttitude("camera", look_at_quat(P_guess, [0,0,0]))
            s.render()
            
            # Récupérer la depth map entière
            depth_map = s.getDepthMap()

            # Filtrer les valeurs finies ET dans (0, ALT_GUESS)
            mask = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < ALT_GUESS)
            valid_depths = depth_map[mask]

            if valid_depths.size == 0:
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue

            # Choisir la médiane
            depth = float(np.median(valid_depths))
            alt_ground = ALT_GUESS - depth

            # Point de surface
            P_surf = geodetic_to_cartesian_BCBF_position(
                         lon, lat, alt_ground, moon_radius, moon_radius)
            
            # Définir un point cible commun (peut être légèrement décalé du point de surface)
            # pour garantir que les deux caméras voient une zone commune
            target_offset = np.random.uniform(-500, 500)  # Décalage du point cible en mètres
            up = normalize(P_surf)
            east = normalize(np.cross([0, 0, 1], up))
            if np.linalg.norm(east) < 1e-16:
                east = np.array([0, 1, 0])
            north = np.cross(up, east)
            
            # Point cible avec petit décalage aléatoire
            target_direction = np.random.uniform(0, 360)
            target_rad = np.deg2rad(target_direction)
            target_offset_vec = target_offset * (np.cos(target_rad) * north + np.sin(target_rad) * east)
            P_target = P_surf + target_offset_vec
            
            # Générer deux configurations de caméra qui regardent vers la même zone
            # Utiliser différents niveaux de contrainte pour varier les configurations
            constraint_types = ["tight", "moderate", "loose"]
            constraint1 = np.random.choice(constraint_types, p=[0.3, 0.5, 0.2])
            constraint2 = np.random.choice(constraint_types, p=[0.3, 0.5, 0.2])
            
            cam1_config = generate_camera_config(P_surf, alt_ground, ALT_BASE, 1, P_target, constraint1)
            cam2_config = generate_camera_config(P_surf, alt_ground, ALT_BASE, 2, P_target, constraint2)
            
            # Vérifier l'infini
            view1_ok, inf_count1 = check_infinity_in_view(s, cam1_config["position"], cam1_config["attitude"])
            view2_ok, inf_count2 = check_infinity_in_view(s, cam2_config["position"], cam2_config["attitude"])

            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Statistiques
            altitude_stats["modes"][cam1_config["mode"]] += 1
            altitude_stats["modes"][cam2_config["mode"]] += 1
            altitude_stats["pitch_ranges"].extend([cam1_config["pitch"], cam2_config["pitch"]])
            altitude_stats["roll_ranges"].extend([cam1_config["roll"], cam2_config["roll"]])
            altitude_stats["baselines"].extend([cam1_config["baseline"], cam2_config["baseline"]])
            altitude_stats["altitude_variations"].extend([cam1_config["alt_variation"], cam2_config["alt_variation"]])
            
            # Calcul de la distance entre les deux caméras
            inter_camera_distance = np.linalg.norm(cam1_config["position"] - cam2_config["position"])
            
            # Affichage
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS} (total: {total_valid_pairs+valid_pairs+1}): "
                      f"Lat={lat:.2f}°, Lon={lon:.1f}°")
                print(f"  Camera 1: {cam1_config['mode']:15s} Alt={cam1_config['altitude']:.0f}m "
                      f"Pitch={cam1_config['pitch']:+.1f}° Offset={cam1_config['angle_offset_from_target']:.1f}° ({constraint1})")
                print(f"  Camera 2: {cam2_config['mode']:15s} Alt={cam2_config['altitude']:.0f}m "
                      f"Pitch={cam2_config['pitch']:+.1f}° Offset={cam2_config['angle_offset_from_target']:.1f}° ({constraint2})")
                print(f"  Distance inter-caméras: {inter_camera_distance:.0f}m")

            # Enregistrer la position
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1

            # Éclairages
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_surf)
                positions.extend([cam1_config["position"], cam2_config["position"]])
                attitudes.extend([cam1_config["attitude"], cam2_config["attitude"]])
                suns.extend([sun_pos, sun_pos])
                
                # Metadata complète pour chaque image
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

        # Résumé pour cette altitude
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude de référence {ALT_BASE}m terminée:")
        print(f"  - Paires générées: {valid_pairs}/{N_PAIRS}")
        print(f"  - Paires rejetées: {rejected_pairs}")
        print(f"  - Taux de succès: {100*valid_pairs/attempts:.1f}%")
        print(f"  - Répartition des modes:")
        total_modes = sum(altitude_stats["modes"].values())
        for mode, count in altitude_stats["modes"].items():
            print(f"    • {mode}: {count} ({100*count/total_modes:.1f}%)")
        print(f"  - Pitch moyen: {np.mean(altitude_stats['pitch_ranges']):.1f}° "
              f"(min: {np.min(altitude_stats['pitch_ranges']):.1f}°, "
              f"max: {np.max(altitude_stats['pitch_ranges']):.1f}°)")
        print(f"  - Baseline moyenne: {np.mean(altitude_stats['baselines']):.0f}m")

    # Export des fichiers
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Trajectoire caméras
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("traj_landing_test.csv", index=False)

    # Trajectoire Soleil
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("sun_landing_test.csv", index=False)
    
    # Metadata détaillée
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("metadata_landing360.csv", index=False)
    
    # Créer un résumé par paire
    pair_summary = []
    for i in range(0, len(metadata), 6):  # 6 images par paire (2 cam × 3 éclairages)
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
    df_summary.to_csv("pair_summary_altitude_distribution_varied_stereo.csv", index=False)

    # ────────────── RAPPORT FINAL ──────────────
    print("\n" + "="*80)
    print("RAPPORT FINAL DE GÉNÉRATION (STEREO VARIÉ)")
    print("="*80)
    
    print(f"\n✅ Fichiers exportés:")
    print("   • traj_altitude_distribution_varied_stereo_90_180.csv")
    print("   • sun_traj_altitude_distribution_varied_stereo_0_90.csv")
    print("   • metadata_altitude_distribution_varied_stereo_0_90.csv")
    print("   • pair_summary_altitude_distribution_varied_stereo.csv")
    
    print(f"\nSTATISTIQUES GLOBALES:")
    print(f"  Total paires générées: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)} ({total_valid_pairs} paires × {N_LIGHTS} éclairages × 2 caméras)")
    print(f"  Total rejets: {total_rejected_pairs}")
    print(f"    - Depth invalide: {rejection_stats['invalid_depth']}")
    print(f"    - Pixels infinis: {rejection_stats['infinity']}")
    print(f"  Taux de succès global: {100*total_valid_pairs/total_attempts:.1f}%")
    
    # Analyse des modes de caméra
    print("\n" + "-"*60)
    print("RÉPARTITION GLOBALE DES MODES DE CAMÉRA:")
    print("-"*60)
    mode_counts = df_meta.groupby('mode').size()
    total_cameras = len(df_meta)
    for mode in CAMERA_MODES:
        count = mode_counts.get(mode, 0)
        percentage = 100 * count / total_cameras
        print(f"  {mode:15s}: {count:5d} ({percentage:5.1f}%)")
    
    # Statistiques des angles
    print("\n" + "-"*60)
    print("STATISTIQUES DES ANGLES:")
    print("-"*60)
    print(f"  Pitch:")
    print(f"    - Moyenne: {df_meta['pitch_deg'].mean():.1f}°")
    print(f"    - Min/Max: {df_meta['pitch_deg'].min():.1f}° / {df_meta['pitch_deg'].max():.1f}°")
    print(f"  Roll:")
    print(f"    - Moyenne: {df_meta['roll_deg'].mean():.1f}°")
    print(f"    - Min/Max: {df_meta['roll_deg'].min():.1f}° / {df_meta['roll_deg'].max():.1f}°")
    
    # Statistiques des distances inter-caméras
    print("\n" + "-"*60)
    print("DISTANCES INTER-CAMÉRAS:")
    print("-"*60)
    print(f"  Moyenne: {df_summary['inter_camera_distance_m'].mean():.0f}m")
    print(f"  Min/Max: {df_summary['inter_camera_distance_m'].min():.0f}m / "
          f"{df_summary['inter_camera_distance_m'].max():.0f}m")
    
    # Combinaisons de modes
    print("\n" + "-"*60)
    print("COMBINAISONS DE MODES LES PLUS FRÉQUENTES:")
    print("-"*60)
    mode_combos = df_summary.groupby(['cam1_mode', 'cam2_mode']).size().sort_values(ascending=False).head(10)
    for (mode1, mode2), count in mode_combos.items():
        percentage = 100 * count / len(df_summary)
        print(f"  {mode1:15s} + {mode2:15s}: {count:4d} ({percentage:5.1f}%)")
    
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
    
    # Récapitulatif par altitude
    print("\n" + "-"*60)
    print("RÉCAPITULATIF PAR ALTITUDE DE RÉFÉRENCE:")
    print("-"*60)
    print(f"{'Alt. Réf (m)':>12} | {'Objectif':>8} | {'Générées':>8} | {'Complété':>8}")
    print("-"*60)
    
    # Grouper par altitude de référence
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
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
    print("Configuration: Paires stéréo avec géométries très variées")
    print("- Modes: nadir, oblique, oblique fort")
    print("- Variations d'altitude: ±30%")
    print("- Variations de roll: ±15°")
    print("- Baseline: 100-3000m dans toutes les directions")
    print("="*80)