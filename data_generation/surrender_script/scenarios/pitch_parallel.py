import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4

# ────────────── CONFIGURATION GLOBALE ──────────────
moon_radius = 1_737_400

# Configuration de la distribution par altitude
TOTAL_PAIRS = 50
ALTITUDE_DISTRIBUTION = [
    {"altitude": 20000, "pairs": 5},
    {"altitude": 23000, "pairs": 5},
    {"altitude": 26000, "pairs": 5},
    {"altitude": 29000, "pairs": 5},
    {"altitude": 32000, "pairs": 5},
    {"altitude": 35000, "pairs": 5},
    {"altitude": 38000, "pairs": 5},
    {"altitude": 41000, "pairs": 5},
    {"altitude": 44000, "pairs": 5},
    {"altitude": 47000, "pairs": 5},
]

# Vérification du total
total_check = sum(alt["pairs"] for alt in ALTITUDE_DISTRIBUTION)
if total_check != TOTAL_PAIRS:
    print(f"⚠️ Attention: Total configuré ({total_check}) != TOTAL_PAIRS ({TOTAL_PAIRS})")
    diff = TOTAL_PAIRS - total_check + ALTITUDE_DISTRIBUTION[-1]["pairs"]
    ALTITUDE_DISTRIBUTION[-1]["pairs"] = diff
    print(f"   Ajusté le dernier palier à {diff} paires")

# BASELINE : Ratio B/H aléatoire pour stéréo avec FOV 30°
B_H_RATIO_MIN = 0.02
B_H_RATIO_MAX = 0.1

# =============== STRATÉGIE DE COUVERTURE GLOBALE ===============
LAT_RANGE = (-85.0, 85.0)    # Éviter les pôles extrêmes (convergence problématique)
LON_RANGE = (0.0, 360.0)

# Paramètres pour filtrer l'infini
MAX_INFINITY_PIXELS = 0
DEPTH_CHECK_RESOLUTION = 128
INFINITY_THRESHOLD = 100_000_00

N_LIGHTS = 3
UA = 149_597_870_700

def generate_sun_setups():
    setups = [
        (150, 160),  # Azimuth: 150°, Incidence: 160°
        (250, 20),   # Azimuth: 250°, Incidence: 20°
        (360, 165)   # Azimuth: 360°, Incidence: 165°
    ]
    return setups

SUN_SETUPS = generate_sun_setups()

# =============== STRATÉGIES D'ÉCHANTILLONNAGE GÉOGRAPHIQUE ===============

def create_adaptive_global_grid(total_pairs):
    """
    Crée une grille adaptative qui tient compte:
    1. De la convergence des méridiens (moins de points près des pôles)
    2. D'une répartition équilibrée en surface réelle
    3. D'un échantillonnage quasi-aléatoire pour éviter les patterns
    """
    # Diviser en bandes de latitude avec densité adaptative
    lat_bands = [
        {"range": (-85, -60), "weight": 0.8, "name": "Pôle Sud"},
        {"range": (-60, -30), "weight": 1.0, "name": "Sud"},
        {"range": (-30, 0),   "weight": 1.0, "name": "Sud équatorial"},
        {"range": (0, 30),    "weight": 1.0, "name": "Nord équatorial"},
        {"range": (30, 60),   "weight": 1.0, "name": "Nord"},
        {"range": (60, 85),   "weight": 0.8, "name": "Pôle Nord"},
    ]
    
    # Calculer la distribution des points par bande
    total_weight = sum(band["weight"] for band in lat_bands)
    points = []
    
    print("\nDistribution géographique prévue:")
    print("-" * 50)
    
    for band in lat_bands:
        # Nombre de points pour cette bande
        n_points = int(total_pairs * band["weight"] / total_weight)
        lat_min, lat_max = band["range"]
        
        print(f"{band['name']:15}: {n_points:3d} points ({lat_min:3.0f}° à {lat_max:3.0f}°)")
        
        # Facteur de réduction pour les longitudes près des pôles
        lat_center = (lat_min + lat_max) / 2
        cos_factor = max(0.3, abs(np.cos(np.deg2rad(lat_center))))
        
        # Génération quasi-uniforme avec jitter
        for i in range(n_points):
            # Échantillonnage stratifié en latitude
            lat_stratified = lat_min + (lat_max - lat_min) * (i + np.random.uniform(0.1, 0.9)) / n_points
            
            # Longitude adaptée à la latitude
            lon_base = np.random.uniform(0, 360)
            
            # Ajouter un petit jitter pour éviter les alignements
            lat_jitter = np.random.uniform(-1, 1) * (lat_max - lat_min) * 0.05
            lon_jitter = np.random.uniform(-5, 5)
            
            final_lat = np.clip(lat_stratified + lat_jitter, lat_min, lat_max)
            final_lon = (lon_base + lon_jitter) % 360
            
            points.append((final_lat, final_lon, band["name"]))
    
    print(f"Total généré: {len(points)} points")
    return points

def create_systematic_global_coverage(total_pairs):
    """
    Alternative: Grille systématique avec échantillonnage aléatoire par cellule
    """
    # Calculer une grille approximative
    target_spacing_deg = np.sqrt(360 * 170 / total_pairs)  # 170° de latitude utile
    
    n_lat_bands = max(6, int(170 / target_spacing_deg))
    
    points = []
    lat_edges = np.linspace(-85, 85, n_lat_bands + 1)
    
    print(f"\nGrille systématique: {n_lat_bands} bandes de latitude")
    print("-" * 50)
    
    for i in range(n_lat_bands):
        lat_min, lat_max = lat_edges[i], lat_edges[i+1]
        lat_center = (lat_min + lat_max) / 2
        
        # Nombre de points longitude adapté à la latitude
        cos_factor = max(0.3, abs(np.cos(np.deg2rad(lat_center))))
        n_lon = max(2, int(total_pairs / n_lat_bands * cos_factor))
        
        # Répartir les longitudes
        for j in range(n_lon):
            # Position de base
            lat = lat_center + np.random.uniform(-0.4, 0.4) * (lat_max - lat_min)
            lon = (360 * j / n_lon + np.random.uniform(-10, 10)) % 360
            
            points.append((lat, lon, f"Band_{i}"))
            if len(points) >= total_pairs:
                break
        
        if len(points) >= total_pairs:
            break
    
    return points[:total_pairs]

def create_random_weighted_coverage(total_pairs):
    """
    Échantillonnage complètement aléatoire mais pondéré par zones
    """
    points = []
    
    # Zones avec poids différents
    zones = [
        {"lat_range": (-85, -45), "weight": 1.0},
        {"lat_range": (-45, -15), "weight": 1.2},
        {"lat_range": (-15, 15),  "weight": 1.3},  # Zone équatoriale plus importante
        {"lat_range": (15, 45),   "weight": 1.2},
        {"lat_range": (45, 85),   "weight": 1.0},
    ]
    
    # Normaliser les poids
    total_weight = sum(z["weight"] for z in zones)
    zone_probs = [z["weight"] / total_weight for z in zones]
    
    for _ in range(total_pairs):
        # Choisir une zone selon les probabilités
        zone_idx = np.random.choice(len(zones), p=zone_probs)
        zone = zones[zone_idx]
        
        # Générer un point dans cette zone
        lat = np.random.uniform(*zone["lat_range"])
        lon = np.random.uniform(0, 360)
        
        points.append((lat, lon, f"Zone_{zone_idx}"))
    
    return points

# ────────────── FONCTIONS UTILITAIRES INCHANGÉES ──────────────

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

def check_infinity_in_view(s, position, attitude):
    """
    Vérifie si la vue contient des pixels à l'infini
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

    # Configuration du soleil et de la lune
    pos_sun_init = vec3(0, 0, UA)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(3e16 * vec4(1,1,1,1))
    s.createBRDF("lambert", "hapke.brdf", 0.12)
    
    # ============ ICI : CHANGER POUR VOTRE DEM GLOBAL ============
    s.createSphericalDEM("moon_dem", "change2_20m.dem", "lambert", "")
    # =============================================================

    # CHOIX DE LA STRATÉGIE D'ÉCHANTILLONNAGE
    print("Stratégies d'échantillonnage disponibles:")
    print("1. Grille adaptative (recommandé)")
    print("2. Grille systématique")
    print("3. Aléatoire pondéré")
    
    # Ici vous pouvez choisir la stratégie
    SAMPLING_STRATEGY = 1  # Changez selon vos besoins
    
    if SAMPLING_STRATEGY == 1:
        print("\n🌍 Utilisation de la grille adaptative")
        grid_points = create_adaptive_global_grid(TOTAL_PAIRS)
    elif SAMPLING_STRATEGY == 2:
        print("\n🌍 Utilisation de la grille systématique")
        grid_points = create_systematic_global_coverage(TOTAL_PAIRS)
    else:
        print("\n🌍 Utilisation de l'échantillonnage aléatoire pondéré")
        grid_points = create_random_weighted_coverage(TOTAL_PAIRS)

    # Mélanger pour éviter les patterns temporels
    np.random.shuffle(grid_points)
    
    print(f"\n✅ {len(grid_points)} points générés pour la couverture globale")

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
            "sun_angles": [],
            "zones": {}
        }

        while valid_pairs < N_PAIRS and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Utiliser les points de la grille pré-générée
            if grid_index < len(grid_points):
                lat, lon, zone = grid_points[grid_index]
                grid_index += 1
            else:
                # Si on a épuisé la grille, continuer aléatoirement
                lat = np.random.uniform(*LAT_RANGE)
                lon = np.random.uniform(*LON_RANGE)
                zone = "Random"
            
            # Mettre à jour les stats de zone
            altitude_stats["zones"][zone] = altitude_stats["zones"].get(zone, 0) + 1

            # Estimation de l'altitude du terrain via depth-map
            ALT_GUESS = 10000
            P_guess = geodetic_to_cartesian_BCBF_position(
                         lon, lat, ALT_GUESS, moon_radius, moon_radius)
            s.setObjectPosition("camera", vec3(*P_guess))
            s.setObjectAttitude("camera", look_at_quat(P_guess, [0,0,0]))
            s.render()
            
            # Récupérer la depth map
            depth_map = s.getDepthMap()
            
            # Filtrer les valeurs valides
            mask = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < ALT_GUESS)
            valid_depths = depth_map[mask]

            if valid_depths.size == 0:
                rejection_stats["invalid_depth"] += 1
                rejected_pairs += 1
                continue

            # Altitude du terrain
            depth = float(np.median(valid_depths))
            alt_ground = ALT_GUESS - depth

            # Positions des caméras
            P_surf = geodetic_to_cartesian_BCBF_position(
                         lon, lat, alt_ground, moon_radius, moon_radius)
            up = normalize(P_surf)
            P1 = P_surf + ALT * up
            forward1 = normalize(P_surf - P1)
            right = normalize(np.cross(forward1, up))
            if np.linalg.norm(right) < 1e-16:
                right = np.array([0,1,0])
            qA = frame2quat(forward1, right)

            # Génération baseline adaptative
            B_H_RATIO = np.random.uniform(B_H_RATIO_MIN, B_H_RATIO_MAX)
            altitude_stats["baseline_ratios"].append(B_H_RATIO)
            BASE_ADAPTIVE = ALT * B_H_RATIO
            
            # Calculs informatifs
            ground_coverage = 2 * ALT * np.tan(np.deg2rad(15))
            convergence_angle = np.rad2deg(np.arctan(BASE_ADAPTIVE / (2 * ALT)))
            
            # Affichage de progression
            if valid_pairs % 25 == 0 or valid_pairs < 5:
                print(f"\nPair {valid_pairs+1}/{N_PAIRS}: "
                      f"Lat={lat:.1f}°, Lon={lon:.1f}° ({zone})")
                print(f"  Baseline={BASE_ADAPTIVE:.0f}m (B/H={B_H_RATIO:.3f})")
            
            # Deuxième caméra
            P2_tan = P1 + np.random.choice([-1,1]) * BASE_ADAPTIVE * right
            P2 = normalize(P2_tan) * (moon_radius + alt_ground + ALT)
            forward2 = normalize(P_surf - P2)
            qB = frame2quat(forward2, right)

            # Vérifier l'infini dans les deux vues
            view1_ok, inf_count1 = check_infinity_in_view(s, P1, qA)
            view2_ok, inf_count2 = check_infinity_in_view(s, P2, qB)

            if not (view1_ok and view2_ok):
                rejection_stats["infinity"] += 1
                rejected_pairs += 1
                continue

            # Enregistrer la couverture
            lat_key = f"{lat:.1f}"
            altitude_stats["lat_coverage"][lat_key] = altitude_stats["lat_coverage"].get(lat_key, 0) + 1

            # Générer les éclairages
            for az, inc in SUN_SETUPS:
                sun_pos = sun_position_local(az, inc, P_surf)
                positions.extend([P1, P2])
                attitudes.extend([qA, qB])
                suns.extend([sun_pos, sun_pos])
                altitude_stats["sun_angles"].append(inc)
                
                # Metadata complète
                metadata.extend([
                    {
                        "pair_id": total_valid_pairs + valid_pairs,
                        "altitude_level": alt_level,
                        "cam": 1, 
                        "lat": lat, 
                        "lon": lon,
                        "zone": zone,
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
                        "zone": zone,
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
            
            if valid_pairs % 50 == 0:
                print(f"\nProgress altitude {ALT}m: {valid_pairs}/{N_PAIRS} pairs")

        # Résumé pour cette altitude
        total_valid_pairs += valid_pairs
        total_rejected_pairs += rejected_pairs
        
        print(f"\n✓ Altitude {ALT}m terminée:")
        print(f"  - Paires générées: {valid_pairs}/{N_PAIRS}")
        print(f"  - Paires rejetées: {rejected_pairs}")
        print(f"  - Taux de succès: {100*valid_pairs/attempts:.1f}%")
        print(f"  - Zones couvertes: {list(altitude_stats['zones'].keys())}")

    # Export des fichiers
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    # Trajectoire caméras
    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
    )
    df_cam.to_csv("traj_global_stereo.csv", index=False)

    # Trajectoire Soleil
    df_sun = pd.DataFrame(
        arr_sun,
        columns=["x_sun(m)", "y_sun(m)", "z_sun(m)"]
    )
    df_sun.to_csv("sun_traj_global.csv", index=False)
    
    # Metadata détaillée
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("metadata_global_stereo.csv", index=False)
    
    # Résumé par paire
    pair_summary = []
    for i in range(0, len(metadata), 6):  # 6 images par paire (2 cam × 3 éclairages)
        pair_data = metadata[i]
        pair_summary.append({
            "pair_id": pair_data["pair_id"],
            "latitude": pair_data["lat"],
            "longitude": pair_data["lon"],
            "zone": pair_data["zone"],
            "altitude_m": pair_data["altitude_m"],
            "baseline_m": pair_data["baseline_m"],
            "b_h_ratio": pair_data["b_h_ratio"],
            "ground_coverage_m": pair_data["ground_coverage_m"],
            "convergence_angle_deg": pair_data["convergence_angle_deg"]
        })
    
    df_summary = pd.DataFrame(pair_summary)
    df_summary.to_csv("pair_summary_global.csv", index=False)

    # ────────────── RAPPORT FINAL AVEC ANALYSE GÉOGRAPHIQUE ──────────────
    print("\n" + "="*80)
    print("RAPPORT FINAL - COUVERTURE GLOBALE")
    print("="*80)
    
    print(f"\n✅ Fichiers exportés:")
    print("   • traj_global_stereo.csv")
    print("   • sun_traj_global.csv")
    print("   • metadata_global_stereo.csv")
    print("   • pair_summary_global.csv")
    
    print(f"\nSTATISTIQUES GLOBALES:")
    print(f"  Total paires générées: {total_valid_pairs}/{TOTAL_PAIRS}")
    print(f"  Total images: {len(positions)}")
    print(f"  Total rejets: {total_rejected_pairs}")
    print(f"  Taux de succès global: {100*total_valid_pairs/total_attempts:.1f}%")
    
    # Analyse de la couverture géographique
    print("\n" + "-"*60)
    print("COUVERTURE GÉOGRAPHIQUE RÉALISÉE:")
    print("-"*60)
    
    # Distribution par latitude
    lat_bins = pd.cut(df_summary['latitude'], bins=8, precision=0)
    lat_distribution = df_summary.groupby(lat_bins).size()
    
    print("\nDistribution par latitude:")
    for interval, count in lat_distribution.items():
        percentage = 100 * count / len(df_summary)
        bar = "█" * int(percentage / 2) + "░" * (25 - int(percentage / 2))
        print(f"{str(interval):20}: {bar} {count:3d} ({percentage:4.1f}%)")
    
    # Distribution par zone si disponible
    if 'zone' in df_summary.columns:
        print("\nDistribution par zone géographique:")
        zone_counts = df_summary['zone'].value_counts()
        for zone, count in zone_counts.items():
            percentage = 100 * count / len(df_summary)
            print(f"{zone:15}: {count:3d} paires ({percentage:4.1f}%)")
    
    print("\n" + "="*80)
    print("🌍 GÉNÉRATION GLOBALE TERMINÉE AVEC SUCCÈS!")
    print("="*80)