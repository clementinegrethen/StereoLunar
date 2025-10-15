import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4

# ────────────── CONFIGURATION GLOBALE ──────────────
moon_radius = 1_737_400
ALT_MIN     = 15_000      # altitude minimum
ALT_MAX     = 20_000      # altitude maximum
BASE        = 25          # séparation latérale entre caméras
N_PAIRS     = 10

# NOUVEAU : Couverture complète du DEM LOLA
LAT_RANGE   = (-90.0, -81.0)  # Couverture complète du pôle sud
LON_RANGE   = (0.0, 360.0)

# Grille pour assurer une couverture uniforme
N_LAT_BINS = 20  # Nombre de bandes de latitude
N_LON_BINS = 40  # Nombre de secteurs de longitude

INC_MIN, INC_MAX = 20, 40

# PARAMÈTRES POUR VUES OBLIQUES
DISTANCE_FROM_TARGET_MIN = 30_000   # distance horizontale minimale du point cible (m)
DISTANCE_FROM_TARGET_MAX = 30_250   # distance horizontale maximale du point cible (m)

# Paramètres pour filtrer l'infini
MAX_INFINITY_PIXELS = 0             # ZÉRO pixel à l'infini accepté
DEPTH_CHECK_RESOLUTION = 128        # Résolution pour le test rapide de depth
INFINITY_THRESHOLD = 100_000        # Au-delà de 100km, on considère comme infini

# STRATÉGIES DE VARIATION D'ALTITUDE - Distribution uniforme
ALTITUDE_STRATEGIES = ["same", "close", "different", "progressive"]
ALTITUDE_STRATEGY_WEIGHTS = [0.25, 0.25, 0.25, 0.25]  # Probabilités égales

N_LIGHTS    = 4
UA          = 149_597_870_700

# Préparer configurations solaires
SUN_SETUPS = []
for az in np.linspace(0, 360, N_LIGHTS, endpoint=False):
    inc = INC_MIN + (INC_MAX - INC_MIN) * 0.5 * (1 + np.sin(np.deg2rad(az)))
    SUN_SETUPS.append((az, inc))

# ────────────── FONCTIONS UTILES ──────────────
def normalize(v):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-16 else v / n

# quaternion à partir de forward et right
def frame2quat(forward, right):
    z = normalize(forward)
    x = normalize(right)
    y = normalize(np.cross(z, x))
    R = np.column_stack((x, y, z))
    q = Rotation.from_matrix(R).as_quat()
    return (q[3], q[0], q[1], q[2])

# calcul position cartésienne d'après lon, lat, alt
def geodetic_to_cartesian_BCBF_position(lon, lat, alt, a, b):
    λ = np.deg2rad(((lon + 180) % 360) - 180)
    φ = np.deg2rad(lat)
    N = a**2 / np.sqrt(a**2*np.cos(φ)**2 + b**2*np.sin(φ)**2)
    X = (N + alt) * np.cos(φ) * np.cos(λ)
    Y = (N + alt) * np.cos(φ) * np.sin(λ)
    Z = ((b**2/a**2) * N + alt) * np.sin(φ)
    return np.array([X, Y, Z], dtype=np.float64)

# calcul de la direction du soleil local
def sun_position_local(az_deg, inc_deg, P_surf, dist=UA):
    az  = np.deg2rad(az_deg)
    inc = np.deg2rad(inc_deg)
    up   = normalize(P_surf)
    east = normalize(np.cross([0,0,1], up))
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0,1,0])
    north = np.cross(up, east)
    dir_local = (np.cos(inc)*(np.cos(az)*north + np.sin(az)*east)
                 + np.sin(inc)*up)
    return dir_local * dist

# NOUVELLE FONCTION : Génération de grille spatiale
def create_spatial_grid(lat_range, lon_range, n_lat, n_lon):
    """
    Crée une grille de points uniformément répartis sur la surface
    en tenant compte de la convergence des méridiens près du pôle
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    # Latitude bins uniformes
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    
    # Pour chaque bande de latitude, ajuster le nombre de points en longitude
    # Plus on est proche du pôle, moins on a besoin de points en longitude
    grid_points = []
    
    for i in range(n_lat):
        lat_center = (lat_edges[i] + lat_edges[i+1]) / 2
        
        # Facteur de réduction basé sur cos(latitude)
        # Mais pas trop de réduction car on est déjà près du pôle
        reduction_factor = max(0.3, abs(np.cos(np.deg2rad(lat_center))))
        n_lon_adjusted = max(8, int(n_lon * reduction_factor))
        
        lon_centers = np.linspace(lon_min, lon_max, n_lon_adjusted, endpoint=False)
        
        for lon in lon_centers:
            # Ajouter un peu de jitter pour éviter une grille trop régulière
            lat_jitter = np.random.uniform(-0.05, 0.05) * (lat_edges[i+1] - lat_edges[i])
            lon_jitter = np.random.uniform(-0.5, 0.5) * 360 / n_lon_adjusted
            
            final_lat = np.clip(lat_center + lat_jitter, lat_min, lat_max)
            final_lon = (lon + lon_jitter) % 360
            
            grid_points.append((final_lat, final_lon))
    
    return grid_points

# Fonction pour générer des altitudes selon différentes stratégies
def generate_altitude_pair(strategy, pair_index, alt_min, alt_max):
    """
    Génère une paire d'altitudes selon la stratégie choisie
    """
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
        # Distribution plus uniforme sur toute la gamme d'altitude
        progress = pair_index / float(N_PAIRS)
        # Oscillation sinusoïdale pour couvrir toute la gamme plusieurs fois
        osc = 0.5 * (1 + np.sin(2 * np.pi * 3 * progress))  # 3 cycles complets
        base_alt = alt_min + (alt_max - alt_min) * osc
        
        alt1 = base_alt + np.random.uniform(-500, 500)
        alt2 = base_alt + np.random.uniform(-500, 500)
        alt1 = np.clip(alt1, alt_min, alt_max)
        alt2 = np.clip(alt2, alt_min, alt_max)
        return alt1, alt2
    
    else:
        # Par défaut, aléatoire indépendant
        return np.random.uniform(alt_min, alt_max), np.random.uniform(alt_min, alt_max)

# Fonction pour créer une vue oblique
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
    horizontal = normalize(view_direction)
    angle_from_horizontal = np.arcsin(np.clip(-np.dot(forward, up_cam), -1, 1))
    
    # Vecteur right
    right = normalize(np.cross(forward, up_cam))
    if np.linalg.norm(right) < 1e-16:
        right = east
    
    return P_cam, frame2quat(forward, right), angle_from_horizontal

# Vérifier si une vue contient de l'infini
def check_infinity_in_view(s, position, attitude):
    """
    Vérifie si la vue contient des pixels à l'infini
    Retourne (True si OK, nombre de pixels infinis)
    """
    # Sauvegarder la taille d'image actuelle
    original_size = s.getImageSize()
    
    # Utiliser une résolution plus basse pour le test
    s.setImageSize(DEPTH_CHECK_RESOLUTION, DEPTH_CHECK_RESOLUTION)
    
    # Positionner la caméra et rendre
    s.setObjectPosition("camera", vec3(*position))
    s.setObjectAttitude("camera", attitude)
    s.render()
    
    # Récupérer la depth map
    depth_map = s.getDepthMap()
    
    # Restaurer la taille originale
    s.setImageSize(original_size[0], original_size[1])
    
    # Compter les pixels à l'infini ou très loin
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

    # Soleil "physique"
    pos_sun_init = vec3(0, 0, UA)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(5e17 * vec4(1,1,1,1))

    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "south5m.dem", "lambert", "")

    # Créer la grille de points pour une couverture uniforme
    print(f"Création d'une grille spatiale pour couvrir uniformément la zone {LAT_RANGE}")
    grid_points = create_spatial_grid(LAT_RANGE, LON_RANGE, N_LAT_BINS, N_LON_BINS)
    np.random.shuffle(grid_points)  # Mélanger pour éviter les patterns
    
    print(f"Nombre de points de grille créés: {len(grid_points)}")

    positions, attitudes, suns = [], [], []
    metadata = []
    valid_pairs = 0
    rejected_pairs = 0
    max_attempts = N_PAIRS * 100
    attempts = 0
    
    # Statistiques
    rejection_stats = {strategy: 0 for strategy in ALTITUDE_STRATEGIES}
    lat_coverage = {i: 0 for i in range(N_LAT_BINS)}
    altitude_histogram = {i: 0 for i in range(10)}  # 10 bins d'altitude
    
    # Index pour parcourir la grille
    grid_index = 0

    while valid_pairs < N_PAIRS and attempts < max_attempts:
        attempts += 1
        
        # Utiliser les points de la grille en priorité
        if grid_index < len(grid_points) and valid_pairs < len(grid_points):
            lat, lon = grid_points[grid_index]
            grid_index += 1
        else:
            # Si on a épuisé la grille, générer aléatoirement
            lon = np.random.uniform(*LON_RANGE)
            lat = np.random.uniform(*LAT_RANGE)

        # altitude initiale pour tester profondeur
        ALT_GUESS = 5000
        P_guess = geodetic_to_cartesian_BCBF_position(lon, lat, ALT_GUESS, moon_radius, moon_radius)
        s.setObjectPosition("camera", vec3(*P_guess))
        s.setObjectAttitude("camera", frame2quat(normalize([0,0,0] - P_guess), [1,0,0]))
        s.render()
        depth = float(s.getDepthMap()[256,256])
        alt_ground = ALT_GUESS - depth
        if not np.isfinite(depth):
            continue

        # Point cible sur la surface
        P_target = geodetic_to_cartesian_BCBF_position(lon, lat, alt_ground, moon_radius, moon_radius)
        up_target = normalize(P_target)
        
        # Calculer les vecteurs de base locaux
        east = normalize(np.cross([0,0,1], up_target))
        if np.linalg.norm(east) < 1e-16:
            east = np.array([0,1,0])
        north = np.cross(up_target, east)
        
        # Choisir une stratégie d'altitude avec les poids définis
        strategy = np.random.choice(ALTITUDE_STRATEGIES, p=ALTITUDE_STRATEGY_WEIGHTS)
        alt1, alt2 = generate_altitude_pair(strategy, valid_pairs, ALT_MIN, ALT_MAX)
        
        # Paramètres de distance
        dist1 = np.random.uniform(DISTANCE_FROM_TARGET_MIN, DISTANCE_FROM_TARGET_MAX)
        dist2 = np.random.uniform(DISTANCE_FROM_TARGET_MIN, DISTANCE_FROM_TARGET_MAX)
        
        # Azimuts pour la stéréo
        azimuth_base = np.random.uniform(0, 360)
        azimuth1 = azimuth_base + np.random.uniform(-10, 10)
        azimuth2 = azimuth_base + np.random.uniform(-10, 10)
        
        # Créer les deux vues stéréo
        offset = BASE * east * 0.5
        P1, q1, angle1 = create_true_oblique_view(P_target + offset, up_target, north, east, dist1, alt1, azimuth1)
        P2, q2, angle2 = create_true_oblique_view(P_target - offset, up_target, north, east, dist2, alt2, azimuth2)
        
        # Vérifier que les deux vues n'ont pas de pixels à l'infini
        view1_ok, inf_count1 = check_infinity_in_view(s, P1, q1)
        view2_ok, inf_count2 = check_infinity_in_view(s, P2, q2)
        
        if not (view1_ok and view2_ok):
            rejected_pairs += 1
            rejection_stats[strategy] += 1
            if rejected_pairs % 20 == 0:
                print(f"  Rejected {rejected_pairs} pairs so far (last: {inf_count1}/{inf_count2} inf pixels)")
            continue
        
        # Statistiques de couverture
        lat_bin = int((lat - LAT_RANGE[0]) / (LAT_RANGE[1] - LAT_RANGE[0]) * N_LAT_BINS)
        lat_bin = np.clip(lat_bin, 0, N_LAT_BINS - 1)
        lat_coverage[lat_bin] += 1
        
        # Statistiques d'altitude
        for alt in [alt1, alt2]:
            alt_bin = int((alt - ALT_MIN) / (ALT_MAX - ALT_MIN) * 10)
            alt_bin = np.clip(alt_bin, 0, 9)
            altitude_histogram[alt_bin] += 1
        
        print(f"\nPair {valid_pairs+1}: Lat={lat:.2f}°, Lon={lon:.1f}°, Strategy={strategy}")
        print(f"  Alt1={alt1:.0f}m, Alt2={alt2:.0f}m, ΔAlt={abs(alt2-alt1):.0f}m")
        print(f"  Angles: {np.rad2deg(angle1):.1f}°, {np.rad2deg(angle2):.1f}°")

        # éclairages
        for az, inc in SUN_SETUPS:
            sun_pos = sun_position_local(az, inc, P_target)
            positions.extend([P1, P2])
            attitudes.extend([q1, q2])
            suns.extend([sun_pos, sun_pos])
            metadata.extend([
                {"pair": valid_pairs, "cam": 1, "lat": lat, "lon": lon, 
                 "alt": alt1, "strategy": strategy, "angle_deg": np.rad2deg(angle1), 
                 "distance_km": dist1/1000},
                {"pair": valid_pairs, "cam": 2, "lat": lat, "lon": lon,
                 "alt": alt2, "strategy": strategy, "angle_deg": np.rad2deg(angle2),
                 "distance_km": dist2/1000}
            ])
        
        valid_pairs += 1
        
        if valid_pairs % 50 == 0:
            print(f"\nProgress: {valid_pairs}/{N_PAIRS} pairs generated ({rejected_pairs} rejected)")
            print("Couverture par bande de latitude:")
            for i, count in lat_coverage.items():
                lat_start = LAT_RANGE[0] + i * (LAT_RANGE[1] - LAT_RANGE[0]) / N_LAT_BINS
                lat_end = LAT_RANGE[0] + (i + 1) * (LAT_RANGE[1] - LAT_RANGE[0]) / N_LAT_BINS
                print(f"  [{lat_start:.1f}° à {lat_end:.1f}°]: {count} paires")

    # Export CSV
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)","y(m)","z(m)","q0","qx","qy","qz"]
    )
    df_cam.to_csv("traj_stereo_oblique_full_coverage.csv", index=False)

    df_sun = pd.DataFrame(arr_sun, columns=["x_sun(m)","y_sun(m)","z_sun(m)"])
    df_sun.to_csv("sun_traj_stereo_oblique_full_coverage.csv", index=False)
    
    # Exporter les metadata avec infos de position
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("metadata_stereo_oblique_full_coverage.csv", index=False)

    print("\n" + "="*60)
    print("✅ Export OK : traj_stereo_oblique_full_coverage.csv")
    print(f"Total images générées: {len(positions)} ({valid_pairs} paires × {N_LIGHTS} éclairages × 2 caméras)")
    print(f"Paires rejetées pour cause d'infini: {rejected_pairs}")
    print(f"Taux de succès: {100*valid_pairs/(valid_pairs+rejected_pairs):.1f}%")
    
    # Rapport de couverture finale
    print("\n" + "-"*40)
    print("RAPPORT DE COUVERTURE SPATIALE:")
    print("-"*40)
    for i, count in lat_coverage.items():
        lat_start = LAT_RANGE[0] + i * (LAT_RANGE[1] - LAT_RANGE[0]) / N_LAT_BINS
        lat_end = LAT_RANGE[0] + (i + 1) * (LAT_RANGE[1] - LAT_RANGE[0]) / N_LAT_BINS
        percentage = 100 * count / valid_pairs
        bar = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
        print(f"[{lat_start:5.1f}° à {lat_end:5.1f}°]: {bar} {count:4d} ({percentage:5.1f}%)")
    
    # Répartition des altitudes
    print("\n" + "-"*40)
    print("DISTRIBUTION DES ALTITUDES:")
    print("-"*40)
    for i, count in altitude_histogram.items():
        alt_start = ALT_MIN + i * (ALT_MAX - ALT_MIN) / 10
        alt_end = ALT_MIN + (i + 1) * (ALT_MAX - ALT_MIN) / 10
        percentage = 100 * count / (2 * valid_pairs)  # 2 caméras par paire
        bar = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
        print(f"[{alt_start:5.0f}m à {alt_end:5.0f}m]: {bar} {count:4d} ({percentage:5.1f}%)")
    
    # Répartition des stratégies
    strategy_counts = df_meta.groupby('strategy')['pair'].nunique()
    print("\n" + "-"*40)
    print("RÉPARTITION DES STRATÉGIES D'ALTITUDE:")
    print("-"*40)
    for strat, count in strategy_counts.items():
        percentage = 100 * count / valid_pairs
        print(f"  {strat:12s}: {count:4d} paires ({percentage:5.1f}%)")
    
    # Statistiques sur les angles
    print(f"\n" + "-"*40)
    print("STATISTIQUES ANGLES DE DÉPRESSION:")
    print("-"*40)
    print(f"  Moyenne: {df_meta['angle_deg'].mean():.1f}°")
    print(f"  Médiane: {df_meta['angle_deg'].median():.1f}°")
    print(f"  Min: {df_meta['angle_deg'].min():.1f}°, Max: {df_meta['angle_deg'].max():.1f}°")
    print(f"  Écart-type: {df_meta['angle_deg'].std():.1f}°")
