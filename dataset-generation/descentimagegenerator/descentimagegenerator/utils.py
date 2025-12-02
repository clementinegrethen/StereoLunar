from typing import Tuple, Union
import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation


def normalize(v: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]):
    """
    Beware, this function behaves differently when norm(v) < 1e-16

    Args:
        v (npt.NDArray): vector to normalize
    """
    n = np.linalg.norm(v)
    if n < 1e-16:
        return v
    else:
        return v / n


def yawpitchroll_to_cartesian_BCBF_attitude(
    yaw_deg: float, pitch_deg: float, roll_deg: float, lon_deg: float, lat_deg: float
) -> Tuple[float, float, float, float]:
    """
    Compute local cartesian attitude quaternion from yaw, pitch, roll and position

    Args:
        yaw_deg (float): yaw angle in degrees
        pitch_deg (float): pitch angle in degrees
        roll_deg (float): roll angle in degrees
        lon_deg (float): longitude
        lat_deg (float): geodetic latitude

    Returns:
        (w, x, y, z) attitude quaternion (SurRender format)
    """
    l = np.deg2rad(lon_deg)
    p = np.deg2rad(lat_deg)

    sin = np.sin
    cos = np.cos

    Rned = np.array([[-sin(p)*cos(l), -sin(l), -cos(p)*cos(l)],
                     [-sin(p)*sin(l), cos(l),  -cos(p)*sin(l)],
                     [cos(p)        ,0,        -sin(p)]])

    # yaw pitch roll rotation matrix
    Rypr = Rotation.from_euler(
        "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
    ).as_matrix()

    att = Rotation.from_matrix(Rned @ Rypr).as_quat()
    att = (
        att[3],
        *att[0:3],
    )  # (x, y, z, w) scipy format to (w, x, y, z) SurRender format
    return att


def test_yawpitchroll_to_cartesian_BCBF_attitude():
    ret = yawpitchroll_to_cartesian_BCBF_attitude(
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,lon_deg=0.0, lat_deg=45.0, pos=(4517590.8, 0, 4487348.41),
    )

    print(ret)


def geodetic_to_cartesian_BCBF_position(
    lon_deg: float, lat_deg: float, alt_m: float, a_m: float, b_m: float
) -> Tuple[float, float, float]:
    """
    Args:
        lon_deg (float): longitude angle in degrees
        lat_deg (float): latitude angle in degrees
        alt_m (float): altitude in metres, 0 => body center
        a_m (float): ellipsoid a (m)
        b_m (float): ellipsoid b (m)

    Returns:
        (x, y, z) BCBF position
    """
    # allows using [-180;180] or [0;360] longitudes
    lambda_ = np.deg2rad(((lon_deg + 180) % 360) - 180)
    phi = np.deg2rad(lat_deg)
    # https://en.wikipedia.org/wiki/Geodetic_coordinates
    N = a_m ** 2 / (
        np.sqrt(a_m ** 2 * np.cos(phi) ** 2.0 + b_m ** 2.0 * np.sin(phi) ** 2.0)
    )
    X = (N + alt_m) * np.cos(phi) * np.cos(lambda_)
    Y = (N + alt_m) * np.cos(phi) * np.sin(lambda_)
    Z = ((b_m ** 2.0 / a_m ** 2.0 * N) + alt_m) * np.sin(phi)
    return (X, Y, Z)


def test_geodetic_to_cartesian_BCBF_position():
    geodetic_to_cartesian_BCBF_position(
        lon_deg=0.0, lat_deg=45.0, alt_m=0, a_m=6378137, b_m=6356752.314,
    )

