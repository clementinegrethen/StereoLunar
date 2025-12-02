import logging
from typing import Tuple, Dict, Type, Optional, List
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation
from descentimagegenerator.logger import get_logger
from descentimagegenerator.utils import (
    normalize,
    yawpitchroll_to_cartesian_BCBF_attitude,
    geodetic_to_cartesian_BCBF_position,
)

Position = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class Trajectory:
    def get_position(self, frame_index: int) -> Position:
        """
        Raises:
            IndexError: if frame_index out of range
        """
        ...

    def get_rotation(self, frame_index: int) -> Quaternion:
        """
        Raises:
            IndexError: if frame index out of range
        """
        ...

    def get_length(self) -> Optional[int]:
        """
        Returns:
            The length of the trajectory such as 0 <= frame_index < `get_length()`
            if the trajectory is infinite returns None
        """
        ...


class ListTrajectory(Trajectory):
    """
    Basic Trajectory initialized using a list of Position and a list of Quaternion

    It is suitable for writing tests or supplying a trajectory without reading from disk
    """
    def __init__(
        self, positions: List[Position], quaternions: List[Quaternion],
    ) -> None:
        super().__init__()
        self.positions = positions
        self.quaternions = quaternions

    def get_position(self, frame_index: int) -> Position:
        return self.positions[frame_index]

    def get_rotation(self, frame_index: int) -> Quaternion:
        return self.quaternions[frame_index]

    def get_length(self) -> int:
        return min(len(self.positions), len(self.quaternions))


class FixedTrajectory(Trajectory):
    """
    Fixed trajectory: always the same position, useful when no sun trajectory is provided
    """
    def __init__(self, position):
        self.position = position

    def get_rotation(self, frame_index: int) -> Tuple[float, float, float, float]:
        return tuple(np.array([1, 0, 0, 0], dtype=np.float32))

    def get_position(self, frame_index: int) -> Tuple[float, float, float]:
        return self.position

    def get_length(self) -> int:
        return -1


class TrajectoryReaderException(Exception):
    """
    Something went wrong while reading a trajectory format
    """
    pass


class UnknownTrajectoryFormatException(Exception):
    """
    No trajectory reader could read the format provided
    """
    pass


class TrajectoryReader:
    """
    Reads a Trajectory from a filepath and using optional additional config
    """
    def read_trajectory(
        self,
        trajectory_filepath: str,
        trajectory_config: Optional[Dict],
        is_sun: bool = False,
        logger: logging.Logger = get_logger("TrajectoryReader"),
    ) -> Trajectory:
        """
        Returns:
            Trajectory
        Raises:
            TrajectoryReaderException: if reading the trajectory fails
            UnknownTrajectoryFormatException: if no reader was found
        """
        ...


class DefaultTrajectoryReader(TrajectoryReader):
    def read_trajectory(
        self,
        trajectory_filepath: str,
        trajectory_config: Optional[Dict],
        is_sun: bool = False,
        logger=get_logger("TrajectoryReader"),
    ) -> Trajectory:

        if trajectory_filepath == "Fixed" and trajectory_config is not None:
            return FixedTrajectory(trajectory_config["position"])

        if not osp.exists(trajectory_filepath):
            logger.warning(
                f"{trajectory_filepath} doesn't exist. TrajectoryReader not valid !"
            )
            raise TrajectoryReaderException(f"{trajectory_filepath} doesn't exist")

        _, ext = osp.splitext(trajectory_filepath)

        ext_2_reader: Dict[str, Type[TrajectoryReader]] = {
            ".csv": CsvTrajectoryReader,
            ".mat": MatlabTrajectoryReader,
            ".npz": NpzTrajectoryReader,
        }

        if ext in ext_2_reader.keys():
            return ext_2_reader[ext]().read_trajectory(
                trajectory_filepath, trajectory_config, is_sun
            )
        else:
            logger.warning(
                f"Unknown file extension: '{ext}' for trajectory. TrajectoryReader not valid !"
            )
            raise UnknownTrajectoryFormatException(
                f"Unknown file extension: '{ext}' for trajectory"
            )


class CsvTrajectoryReader(TrajectoryReader):
    def read_trajectory(
        self,
        trajectory_filepath: str,
        trajectory_config: Optional[Dict],
        is_sun: bool = False,
        logger: logging.Logger = get_logger("CSVTrajectoryReader"),
    ) -> Trajectory:
        import pandas as pd

        class CSVTrajectory(Trajectory):
            def __init__(
                self,
                traj: pd.DataFrame,
                header: List[str],
                traj_type: str,
                a: Optional[float] = None,
                b: Optional[float] = None,
            ) -> None:
                self.traj = traj
                self.header = header
                self.traj_type = traj_type
                self.a = a
                self.b = b

            def get_position(self, frame_index: int) -> Tuple[float, float, float]:
                if self.traj_type == "geodetic":
                    lla = self.traj.loc[frame_index, self.header[0:3]].to_numpy()
                    return np.array(geodetic_to_cartesian_BCBF_position(*lla, self.a, self.b))
                else:
                    return self.traj.loc[frame_index, self.header[0:3]].to_numpy()

            def get_rotation(
                self, frame_index: int
            ) -> Tuple[float, float, float, float]:
                if self.traj_type == "geodetic":
                    ypr = self.traj.loc[frame_index, self.header[3:7]].to_numpy()
                    lonlat = self.traj.loc[frame_index, self.header[0:2]].to_numpy()
                    return yawpitchroll_to_cartesian_BCBF_attitude(*ypr, *lonlat)
                else:
                    return self.traj.loc[frame_index, self.header[3:7]].to_numpy()

            def get_length(self) -> int:
                return len(self.traj)

        traj = pd.read_csv(trajectory_filepath, skipinitialspace=True)

        # default traj type
        traj_type = "BCBF"

        if trajectory_config is None:
            raise TrajectoryReaderException(
                "cannot read a CSV trajectory without supplying a trajectory configuration file"
            )

        # header by default (BCBF)
        header = ["x(m)", "y(m)", "z(m)", "q0", "qx", "qy", "qz"]
        if "type" in trajectory_config:
            if trajectory_config["type"] == "geodetic":
                header = ["lon(°)", "lat(°)", "alt(m)", "yaw(°)", "pitch(°)", "roll(°)"]
                traj_type = trajectory_config["type"]
                assert "semi_major_axis" in trajectory_config
                assert "semi_minor_axis" in trajectory_config
                a = trajectory_config["semi_major_axis"]  # semi major axis
                b = trajectory_config["semi_minor_axis"]  # semi minor axis
                return CSVTrajectory(traj, header, traj_type, a, b)
            elif trajectory_config["type"] == "BCBF":
                traj_type="BCBF"
                return CSVTrajectory(traj, header, traj_type)
            else:
                raise UnknownTrajectoryFormatException(
                    "unknown traj type in trajectory config"
                )
        if is_sun:
            header = ["x_sun(m)", "y_sun(m)", "z_sun(m)"]

        return CSVTrajectory(traj, header, traj_type)


class MatlabTrajectoryReader(TrajectoryReader):
    def read_trajectory(
        self,
        trajectory_filepath: str,
        trajectory_config: Optional[Dict],
        is_sun: bool = False,
        logger: logging.Logger = get_logger("MatlabTrajectoryReader"),
    ) -> Trajectory:
        class MatlabTrajectory(Trajectory):
            def __init__(
                self, traj, pos_col, is_sun: bool, sun_index: Optional[int]
            ) -> None:
                self.traj = traj
                self.pos_col = pos_col
                self.is_sun = is_sun
                self.sun_index = sun_index

            def get_position(self, frame_index: int) -> Tuple[float, float, float]:
                if self.is_sun:
                    return self.traj[self.pos_col][self.sun_index]
                else:
                    return self.traj[self.pos_col][frame_index]

            def get_rotation(
                self, frame_index: int
            ) -> Tuple[float, float, float, float]:
                return self.traj[f"q_S{self.traj['cameraNumber'][frame_index][0]}M"][
                    frame_index
                ]

            def get_length(self) -> int:
                return len(self.traj[self.pos_col])

        from scipy.io import loadmat

        if is_sun:
            if trajectory_config is not None and "sun_index" in trajectory_config:
                sun_index = trajectory_config["sun_index"]
            else:
                logger.error(
                    "missing sun index, please supply a 'trajectory_config' with 'sun_index' value"
                )
                raise TrajectoryReaderException("missing sun index")
            pos_col = "posSun_M"
        else:
            sun_index = None
            pos_col = "posSc_M"

        logger.debug(f"loading {trajectory_filepath}")
        traj = loadmat(trajectory_filepath)

        return MatlabTrajectory(traj, pos_col, is_sun, sun_index)


class NpzTrajectoryReader(TrajectoryReader):

    """

    Npz file containing T_CAM(Nx3), R_CAM(N,3,3), T_SUN(N,3) in moon centered frame

    """

    def read_trajectory(
        self,
        trajectory_filepath: str,
        trajectory_config: Optional[Dict],
        is_sun: bool = False,
        logger: logging.Logger = get_logger("NPZTrajectoryReader"),
    ) -> Trajectory:
        class NpzTrajectory(Trajectory):
            def __init__(self, data, t_col: str) -> None:
                self.data = data
                self.t_col = t_col

            def get_length(self) -> int:
                return self.data["R_CAM"].shape[0]

            def get_position(self, frame_index: int) -> Tuple[float, float, float]:
                return self.data[self.t_col][frame_index, ::]

            def get_rotation(
                self, frame_index: int
            ) -> Tuple[float, float, float, float]:
                r = self.data["R_CAM"][frame_index, ::]

                qx, qy, qz, q0 = Rotation.from_matrix(r).as_quat()

                return tuple(np.asarray([q0, qx, qy, qz]))

        data = np.load(trajectory_filepath)
        t_col = "T_SUN" if is_sun else "T_CAM"
        return NpzTrajectory(data, t_col)
