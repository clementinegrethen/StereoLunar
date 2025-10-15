import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from surrender.surrender_client import surrender_client
from surrender.geometry import vec3, vec4

# ────────────── CONFIGURATION GLOBALE ──────────────
moon_radius = 1_737_400
ALT_MIN     = 20_000      # altitude minimum for camera 1
ALT_MAX     = 30_000      # altitude maximum for camera 2
BASE        = 250
N_PAIRS     = 300
LAT_RANGE   = (-90.0, -88)
LON_RANGE   = (  0.0, 360.0)
INC_MIN, INC_MAX           = 20, 40
PITCH_OFF_MIN, PITCH_OFF_MAX = 5, 20  # décalage en pitch pour caméra 2 (°)
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

# quaternion pour orienter la caméra vers une cible
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

# ────────────── SCRIPT PRINCIPAL ──────────────
if __name__ == "__main__":
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer("127.0.0.1")
    s.closeViewer()
    s.setImageSize(512, 512)
    s.setCameraFOVDeg(45, 45)
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.enableRaytracing(True)
    s.setNbSamplesPerPixel(16)
    s.enableLOSmapping(True)
    # Soleil "physique"
    pos_sun_init = vec3(0, 0, UA)
    s.createBRDF("sun", "sun.brdf", {})
    s.createShape("sun", "sphere.shp", {"radius": 696_342_000})
    s.createBody("sun", "sun", "sun", [])
    s.setObjectPosition("sun", pos_sun_init)
    s.setSunPower(5e17 * vec4(1,1,1,1))

    s.createBRDF("lambert", "hapke.brdf", 0.12)
    s.createSphericalDEM("moon_dem", "south5m.dem", "lambert", "")

    positions, attitudes, suns = [], [], []
    valid_pairs   = 0
    max_attempts  = N_PAIRS * 20
    attempts      = 0

    while valid_pairs < N_PAIRS and attempts < max_attempts:
        attempts += 1
        lon = np.random.uniform(*LON_RANGE)
        lat = np.random.uniform(*LAT_RANGE)

        # altitude initiale pour testerprofondeur
        ALT_GUESS = 5000
        P_guess = geodetic_to_cartesian_BCBF_position(lon, lat, ALT_GUESS, moon_radius, moon_radius)
        s.setObjectPosition("camera", vec3(*P_guess))
        s.setObjectAttitude("camera", look_at_quat(P_guess, [0,0,0]))
        s.render()
        depth = float(s.getDepthMap()[256,256])
        print(depth)
        alt_ground = ALT_GUESS - depth
        if not np.isfinite(depth):
            print("euh")
            continue

        P_surf = geodetic_to_cartesian_BCBF_position(lon, lat, alt_ground, moon_radius, moon_radius)
        up     = normalize(P_surf)

        # caméra A à altitude fixe
        P1 = P_surf + ALT_MIN * up
        forward1 = normalize(P_surf - P1)
        right    = normalize(np.cross(forward1, up))
        if np.linalg.norm(right) < 1e-16:
            right = np.array([0,1,0])
        qA = frame2quat(forward1, right)

        # altitude et pitch aléatoires pour caméra B
        alt2 = np.random.uniform(ALT_MIN, ALT_MAX)
        sign = np.random.choice([-1,1])
        P2_nominal = P_surf + alt2*up + BASE*right*sign
        P2 = normalize(P2_nominal) * (moon_radius + alt_ground + alt2)
        forward2 = normalize(P_surf - P2)

        # appliquer décalage pitch
        pitch_off_deg = np.random.uniform(PITCH_OFF_MIN, PITCH_OFF_MAX)
        pitch_off_rad = np.deg2rad(pitch_off_deg)
        forward2_rot = Rotation.from_rotvec(right * pitch_off_rad).apply(forward2)
        qB = frame2quat(forward2_rot, right)

        # éclairages
        for az, inc in SUN_SETUPS:
            sun_pos = sun_position_local(az, inc, P_surf)
            positions.extend([P1, P2])
            attitudes.extend([qA, qB])
            suns.extend([sun_pos, sun_pos])
        valid_pairs += 1

    # export CSV
    arr_pos = np.array(positions)
    arr_att = np.array(attitudes)
    arr_sun = np.array(suns)

    df_cam = pd.DataFrame(
        np.hstack((arr_pos, arr_att)),
        columns=["x(m)","y(m)","z(m)","q0","qx","qy","qz"]
    )
    df_cam.to_csv("traj_stereo_pairs2.csv", index=False)

    df_sun = pd.DataFrame(arr_sun, columns=["x_sun(m)","y_sun(m)","z_sun(m)"])
    df_sun.to_csv("sun_traj_stereo_pairs2.csv", index=False)

    print("✅ Export OK : traj_stereo_pairs2.csv, sun_traj_stereo_pairs2.csv")