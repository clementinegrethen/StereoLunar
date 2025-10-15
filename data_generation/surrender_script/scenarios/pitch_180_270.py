import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4

# ────────────── CONFIGURATION GLOBALE ──────────────
moon_radius = 1_737_400

# NOUVEAU : Configuration de la distribution par altitude
TOTAL_PAIRS = 100  # Nombre total de paires à générer

# Définir les paliers d'altitude et leur répartition
ALTITUDE_DISTRIBUTION = [
    {"altitude": 3500,  "pairs": 10},   # 10%
    {"altitude": 6500,  "pairs": 10},   # 10%
    {"altitude": 9500,  "pairs": 10},   # 10%
    {"altitude": 12500, "pairs": 10},   # 10%
    {"altitude": 15500, "pairs": 10},   # 10%
    {"altitude": 18500, "pairs": 10},   # 10%
    {"altitude": 21500, "pairs": 10},   # 10%
    {"altitude": 24500, "pairs": 10},   # 10%
    {"altitude": 27500, "pairs": 10},   # 10%
    {"altitude": 30500, "pairs": 10},   # 10%
]

# Vérifier que le total correspond
total_check = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)
if total_check != TOTAL_PAIRS:
    print(f"⚠️ Attention: Total configuré ({total_check}) != TOTAL_PAIRS ({TOTAL_PAIRS})")
    # Ajuster automatiquement le dernier palier
    diff = TOTAL_PAIRS - total_check + ALTITUDE_DISTRIBUTION[-1]["pairs"]
    ALTITUDE_DISTRIBUTION[-1]["pairs"] = diff
    print(f"   Ajusté le dernier palier à {diff} paires")

# PARAMÈTRES POUR VUES OBLIQUES - ADAPTATIFS À L'ALTITUDE
# Ratio distance/altitude pour maintenir des angles cohérents
DISTANCE_TO_ALTITUDE_RATIO_MIN = 0.5  # Distance = 1.5 × altitude (angle ~33°)
DISTANCE_TO_ALTITUDE_RATIO_MAX = 1.1  # Distance = 2.0 × altitude (angle ~26°)
BASE = 25  # séparation latérale entre caméras

# STRATÉGIES DE VARIATION D'ALTITUDE
ALTITUDE_STRATEGIES = ["same", "close", "different", "progressive"]
ALTITUDE_STRATEGY_WEIGHTS = [0.25, 0.25, 0.25, 0.25]  # Probabilités égales

# Couverture complète du DEM LOLA
LAT_RANGE   = (-89.9, -87.1)  # Couverture complète du pôle sud
LON_RANGE   = (0.0, 360.0)

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
        (250, 20),   # Azimuth: 135°, Incidence: 150°
        (360, 165)   # Azimuth: 225°, Incidence: 165°
    ]
    return setups

SUN_SETUPS = generate_sun_setups()
print(f"Sun setups générés : {SUN_SETUPS}")

# ────────────── OUTILS ──────────────
def normalize(v):
    v = np.array(v, dtype=np.float64)
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

# NOUVELLES FONCTIONS pour les vues obliques
def generate_altitude_pair(strategy, pair_index, alt_base, alt_range=2000):
    """
    Génère une paire d'altitudes selon la stratégie choisie
    autour d'une altitude de base donnée
    """
    alt_min = max(1000, alt_base - alt_range//2)
    alt_max = alt_base + alt_range//2
    
    if strategy == "same":
        # Même altitude pour les deux caméras
        alt = np.random.uniform(alt_min, alt_max)
        return alt, alt
    
    elif strategy == "close":
        # Altitudes proches (±500-1000m)
        alt1 = np.random.uniform(alt_min, alt_max)
        delta = np.random.uniform(500, 1000) * np.random.choice([-1, 1])
        alt2 = np.clip(alt1 + delta, alt_min, alt_max)
        return alt1, alt2
    
    elif strategy == "different":
        # Une haute, une basse
        if np.random.random() < 0.5:
            alt1 = np.random.uniform(alt_min, alt_min + (alt_max - alt_min) * 0.3)
            alt2 = np.random.uniform(alt_max - (alt_max - alt_min) * 0.3, alt_max)
        else:
            alt1 = np.random.uniform(alt_max - (alt_max - alt_min) * 0.3, alt_max)
            alt2 = np.random.uniform(alt_min, alt_min + (alt_max - alt_min) * 0.3)
        return alt1, alt2
    
    elif strategy == "progressive":
        # Variation progressive basée sur l'index de la paire
        progress = pair_index / 100.0  # Normaliser sur 100 paires
        osc = 0.5 * (1 + np.sin(2 * np.pi * 3 * progress))
        base_alt = alt_min + (alt_max - alt_min) * osc
        
        alt1 = base_alt + np.random.uniform(-500, 500)
        alt2 = base_alt + np.random.uniform(-500, 500)
        alt1 = np.clip(alt1, alt_min, alt_max)
        alt2 = np.clip(alt2, alt_min, alt_max)
        return alt1, alt2
    
    else:
        # Par défaut, aléatoire indépendant
        return np.random.uniform(alt_min, alt_max), np.random.uniform(alt_min, alt_max)

def create_true_oblique_view(P_target, up_target, north, east, distance_horizontal, alt_cam, azimuth_deg):
    """
    Crée une caméra loin du point cible pour une vue vraiment oblique
    """
    # Direction d'où on regarde
    az_rad = np.deg2rad(azimuth_deg)
    view_direction = -(np.cos(az_rad) * north + np.sin(az_rad) * east)
    
    # Position de la caméra : loin du point cible dans la direction opposée
    P_cam_surface = P_target + distance_horizontal * view_direction
    
    # Normaliser pour rester sur la sphère et ajouter l'altitude
    P_cam_surface = normalize(P_cam_surface) * moon_radius
    up_cam = normalize(P_cam_surface)
    P_cam = P_cam_surface + alt_cam * up_cam
    
    # Vecteur forward : de la caméra vers le point cible
    forward = normalize(P_target - P_cam)
    
    # Calculer l'angle de dépression/élévation pour info
    angle_from_horizontal = np.arcsin(np.clip(-np.dot(forward, up_cam), -1, 1))
    
    # Vecteur right
    right = normalize(np.cross(forward, up_cam))
    if np.linalg.norm(right) < 1e-16:
        right = east
    
    return P_cam, frame2quat(forward, right), angle_from_horizontal

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
    s.setCameraFOVDeg(30, 30)
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.enableRaytracing(True)
    s.enableLOSmapping(True)
    s.setNbSamplesPerPixel(16)

    # Configuration du soleil
    pos_sun_init = vec3(0, 0, UA)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(3e16 * vec4(1,1,1,1))
    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "south5m.dem", "lambert", "")

    # Créer la grille de points pour une couverture uniforme
    print(f"Création d'une grille spatiale pour couvrir uniformément la zone {LAT_RANGE}")
    grid_points = create_spatial_grid(LAT_RANGE, LON_RANGE, N_LAT_BINS, N_LON_BINS)
    grid_points.sort(key=lambda p: abs(p[0] + 90))
    print(f"Nombre de points de grille créés: {len(grid_points)}")

    # Afficher le plan de génération
    print("\n" + "="*60)
    print("PLAN DE GÉNÉRATION PAR ALTITUDE (AVEC VUES OBLIQUES):")
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
    strategy_stats = {strategy: 0 for strategy in ALTITUDE_STRATEGIES}
    
    # Index global pour la grille
    grid_index = 0
    
    # Traiter chaque palier d'altitude
    for alt_level, alt_config in enumerate(ALTITUDE_DISTRIBUTION):
        ALT_BASE = alt_config["altitude"]
        N_PAIRS = alt_config["pairs"]
        
        print(f"\n{'='*60}")
        print(f"Génération pour altitude {ALT_BASE}m ({N_PAIRS} paires) - VUES OBLIQUES")
        print(f"{'='*60}")
        
        valid_pairs = 0
        rejected_pairs = 0
        max_attempts = N_PAIRS * 50
        attempts = 0
        
        # Statistiques par altitude
        altitude_stats = {
            "lat_coverage": {},
            "strategies": {strategy: 0 for strategy in ALTITUDE_STRATEGIES},
            "oblique_angles": [],
            "distances": []
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

            # Altitude terrain (depth-map)
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, moon_radius, moon_radius)
            s.setObjectPosition("camera", vec3(*P_guess))
            s.setObjectAttitude("camera", frame2quat(normalize([0,0,0] - P_guess), [1,0,0]))
            s.render()
            
            # Récupérer la depth map entière
            depth_map = s.getDepthMap()
            mask = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < ALT_GUESS)
            valid_depths = depth_map[mask]

            if valid_depths.size == 0:
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue

            depth = float(np.median(valid_depths))
            alt_ground = ALT_GUESS - depth

            # Point cible sur la surface
            P_target = geodetic_to_cartesian_BCBF_position(lon, lat, alt_ground, moon_radius, moon_radius)
            up_target = normalize(P_target)
            
            # Calculer les vecteurs de base locaux
            east = normalize(np.cross([0,0,1], up_target))
            if np.linalg.norm(east) < 1e-16:
                east = np.array([0,1,0])
            north = np.cross(up_target, east)
            
            # Choisir une stratégie d'altitude
            strategy = np.random.choice(ALTITUDE_STRATEGIES, p=ALTITUDE_STRATEGY_WEIGHTS)
            alt1, alt2 = generate_altitude_pair(strategy, valid_pairs, ALT_BASE)
            
            # Paramètres de distance adaptatifs à l'altitude
            # Plus l'altitude est faible, plus la distance doit être faible pour garder un angle cohérent
            ratio1 = np.random.uniform(DISTANCE_TO_ALTITUDE_RATIO_MIN, DISTANCE_TO_ALTITUDE_RATIO_MAX)
            ratio2 = np.random.uniform(DISTANCE_TO_ALTITUDE_RATIO_MIN, DISTANCE_TO_ALTITUDE_RATIO_MAX)
            dist1 = alt1 * ratio1
            dist2 = alt2 * ratio2
            
            # Azimuts pour la stéréo
            azimuth_base = np.random.uniform(0, 360)
            azimuth1 = azimuth_base + np.random.uniform(-10, 10)
            azimuth2 = azimuth_base + np.random.uniform(-10, 10)
            
            # Créer les deux vues obliques stéréo
            offset = BASE * east * 0.5
            P1, q1, angle1 = create_true_oblique_view(P_target + offset, up_target, north, east, dist1, alt1, azimuth1)
            P2, q2, angle2 = create_true_oblique_view(P_target - offset, up_target, north, east, dist2, alt2, azimuth2)

            # Vérifier l'infini
            view1_ok, inf_count1 = check_infinity_in_view(s, P1, q1)
            view2_ok, inf_count2 = check_infinity_in_view(s, P2, q2)
            print(inf_count1)
            print(inf_count2)
            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Enregistrer les statistiques
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1
            altitude_stats["strategies"][strategy] += 1
            altitude_stats["oblique_angles"].extend([np.rad2deg(angle1), np.rad2deg(angle2)])
            altitude_stats["distances"].extend([dist1/1000, dist2/1000])
            strategy_stats[strategy] += 1

            # Affichage périodique
            if valid_pairs % 50 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS} (total: {total_valid_pairs+valid_pairs+1}): "
                      f"Lat={lat:.2f}°, Lon={lon:.1f}°")
                print(f"  Strategy={strategy}, Alt1={alt1:.0f}m, Alt2={alt2:.0f}m (ΔAlt={abs(alt2-alt1):.0f}m)")
                print(f"  Angles obliques: {np.rad2deg(angle1):.1f}°, {np.rad2deg(angle2):.1f}°")
                print(f"  Distances: {dist1/1000:.1f}km, {dist2/1000:.1f}km (ratios: {dist1/alt1:.1f}×, {dist2/alt2:.1f}×)")
                print(f"  Angles théoriques: {np.rad2deg(np.arctan(alt1/dist1)):.1f}°, {np.rad2deg(np.arctan(alt2/dist2)):.1f}°")

            # Enregistrer les éclairages
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_target)
                positions.extend([P1, P2])
                attitudes.extend([q1, q2])
                suns.extend([sun_pos, sun_pos])
                
                # Metadata complète pour chaque image
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

        # Résumé pour cette altitude
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude {ALT_BASE}m terminée:")
        print(f"  - Paires générées: {valid_pairs}/{N_PAIRS}")
        print(f"  - Paires rejetées: {rejected_pairs}")
        print(f"  - Taux de succès: {100*valid_pairs/attempts:.1f}%")
        if altitude_stats["oblique_angles"]:
            print(f"  - Angle oblique moyen: {np.mean(altitude_stats['oblique_angles']):.1f}°")
            print(f"  - Distance moyenne: {np.mean(altitude_stats['distances']):.1f}km")

    # Export des fichiers
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Trajectoire caméras
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("traj_oblique_test.csv", index=False)

    # Trajectoire Soleil
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("sun_oblique_test.csv", index=False)
    
    # Metadata détaillée
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("meta_oblique_test.csv", index=False)
    
    # Créer un résumé par paire
    pair_summary = []
    for i in range(0, len(metadata), 6):  # 6 images par paire (2 cam × 3 éclairages)
        pair_data = metadata[i]
        pair_summary.append({
            "pair_id": pair_data["pair_id"],
            "latitude": pair_data["lat"],
            "longitude": pair_data["lon"],
            "altitude_strategy": pair_data["altitude_strategy"],
            "alt1_m": pair_data["altitude_m"],
            "alt2_m": metadata[i+1]["altitude_m"],  # Deuxième caméra
            "target_altitude_m": pair_data["target_altitude_m"],
            "distance1_km": pair_data["distance_km"],
            "distance2_km": metadata[i+1]["distance_km"],
            "distance_ratio1": pair_data["distance_ratio"],
            "distance_ratio2": metadata[i+1]["distance_ratio"],
            "oblique_angle1_deg": pair_data["oblique_angle_deg"],
            "oblique_angle2_deg": metadata[i+1]["oblique_angle_deg"]
        })
    
    df_summary = pd.DataFrame(pair_summary)
    df_summary.to_csv("pair_summary_test_bolqieu.csv", index=False)

    # ────────────── RAPPORT FINAL ──────────────
    print("\n" + "="*80)
    print("RAPPORT FINAL DE GÉNÉRATION - VUES OBLIQUES PAR ALTITUDE")
    print("="*80)
    
    print(f"\n✅ Fichiers exportés:")
    print("   • traj_stereo_oblique_altitude_distribution.csv")
    print("   • sun_traj_oblique_altitude_distribution.csv")
    print("   • metadata_oblique_altitude_distribution.csv")
    print("   • pair_summary_oblique_altitude_distribution.csv")
    
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
    
    altitude_counts = df_summary.groupby('target_altitude_m').size()
    for alt_config in ALTITUDE_DISTRIBUTION:
        alt = alt_config["altitude"]
        target = alt_config["pairs"]
        actual = altitude_counts.get(alt, 0)
        completion = 100 * actual / target if target > 0 else 0
        print(f"{alt:>12} | {target:>8} | {actual:>8} | {completion:>7.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL':>12} | {TOTAL_PAIRS:>8} | {total_valid_pairs:>8} | {100*total_valid_pairs/TOTAL_PAIRS:>7.1f}%")
    
    # Répartition des stratégies
    print("\n" + "-"*60)
    print("RÉPARTITION DES STRATÉGIES D'ALTITUDE:")
    print("-"*60)
    for strategy, count in strategy_stats.items():
        percentage = 100 * count / total_valid_pairs if total_valid_pairs > 0 else 0
        print(f"  {strategy:12s}: {count:4d} paires ({percentage:5.1f}%)")
    
    # Statistiques des angles obliques
    print("\n" + "-"*60)
    print("STATISTIQUES DES ANGLES OBLIQUES:")
    print("-"*60)
    angles = df_meta['oblique_angle_deg']
    print(f"  Moyenne: {angles.mean():.1f}°")
    print(f"  Médiane: {angles.median():.1f}°")
    print(f"  Min: {angles.min():.1f}°, Max: {angles.max():.1f}°")
    print(f"  Écart-type: {angles.std():.1f}°")
    
    # Statistiques des distances et ratios
    print(f"\n  Distances et ratios:")
    distances = df_meta['distance_km']
    ratios = df_meta['distance_ratio']
    print(f"  Distance moyenne: {distances.mean():.1f}km")
    print(f"  Min: {distances.min():.1f}km, Max: {distances.max():.1f}km")
    print(f"  Ratio distance/altitude moyen: {ratios.mean():.1f}×")
    print(f"  Ratio min: {ratios.min():.1f}×, max: {ratios.max():.1f}×")
    
    print("\n" + "="*80)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
    print("Les caméras utilisent des vues obliques avec distances adaptatives à l'altitude")
    print(f"Ratios distance/altitude: {DISTANCE_TO_ALTITUDE_RATIO_MIN:.1f}× à {DISTANCE_TO_ALTITUDE_RATIO_MAX:.1f}×")
    print("Angles de vue maintenus entre ~26° et ~33° selon l'altitude")
    print("="*80)