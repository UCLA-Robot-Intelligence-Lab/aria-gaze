from __future__ import annotations
import projectaria_tools.core.calibration
import projectaria_tools.core.sensor_data
import datetime
import numpy
import typing
from . import hand_tracking
__all__ = ['ClosedLoopTrajectoryPose', 'EyeGaze', 'EyeGazeVergence', 'GlobalPointPosition', 'HandTrackingDataPaths', 'MpsDataPaths', 'MpsDataPathsProvider', 'MpsDataProvider', 'MpsEyegazeDataPaths', 'MpsSlamDataPaths', 'OnlineCalibration', 'OpenLoopTrajectoryPose', 'PointObservation', 'StaticCameraCalibration', 'StreamCompressionMode', 'compute_depth_and_combined_gaze_direction', 'get_eyegaze_point_at_depth', 'get_gaze_intersection_point', 'get_gaze_vectors', 'hand_tracking', 'read_closed_loop_trajectory', 'read_eyegaze', 'read_global_point_cloud', 'read_online_calibration', 'read_open_loop_trajectory', 'read_point_observations', 'read_static_camera_calibrations']
class ClosedLoopTrajectoryPose:
    """
    
              Closed loop trajectory is the pose estimation output by our mapping process, in an arbitrary
      gravity aligned world coordinate frame. The estimation includes pose and dynamics (translational
      and angular velocities).
    
      Closed loop trajectories are fully bundle adjusted with detected loop closures, reducing the VIO
      drift which is present in the open loop trajectories. However, due to the loop closure
      correction, the “relative” and “local” trajectory accuracy within a short time span (i.e.
      seconds) might be worse compared to open loop trajectories.
    
      In some datasets we also share and use this format for trajectory pose ground truth from
      simulation or Optitrack
        
    """
    angular_velocity_device: numpy.ndarray[numpy.float64[3, 1]]
    device_linear_velocity_device: numpy.ndarray[numpy.float64[3, 1]]
    graph_uid: str
    gravity_world: numpy.ndarray[numpy.float64[3, 1]]
    quality_score: float
    tracking_timestamp: datetime.timedelta
    transform_world_device: SE3
    utc_timestamp: datetime.timedelta
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class EyeGaze:
    """
    An object representing single Eye gaze output.
    """
    depth: float
    pitch: float
    pitch_high: float
    pitch_low: float
    session_uid: str
    tracking_timestamp: datetime.timedelta
    vergence: EyeGazeVergence
    yaw: float
    yaw_high: float
    yaw_low: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class EyeGazeVergence:
    left_yaw: float
    left_yaw_high: float
    left_yaw_low: float
    right_yaw: float
    right_yaw_high: float
    right_yaw_low: float
    tx_left_eye: float
    tx_right_eye: float
    ty_left_eye: float
    ty_right_eye: float
    tz_left_eye: float
    tz_right_eye: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class GlobalPointPosition:
    distance_std: float
    graph_uid: str
    inverse_distance_std: float
    position_world: numpy.ndarray[numpy.float64[3, 1]]
    uid: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class HandTrackingDataPaths:
    """
    A struct that includes the file paths of all MPS Hand Tracking data for a VRS sequence processed by MPS.
    """
    summary: str
    wrist_and_palm_poses: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class MpsDataPaths:
    """
    A struct that includes the file paths of all MPS data for a sequence.
    """
    eyegaze: ...
    hand_tracking: ...
    root: str
    slam: ...
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class MpsDataPathsProvider:
    """
    This class is allows you to get all MPS data paths associated with an Aria sequence. 
    Note that all Aria open datasets will have MPS results which fit the format specified in this data provider.
    Use this data provider to avoid breaking changes in your code due to changes in MPS files
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str) -> None:
        ...
    def get_data_paths(self) -> MpsDataPaths:
        """
        Get the resulting data paths
        """
class MpsDataProvider:
    """
    This class is to load all MPS data given an MpsDataPaths object, and also provide all API needed to query that data. NOTE: to minimize disk usage, this data provider only loads data from disk after that data type is first queried.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: MpsDataPaths) -> None:
        ...
    def get_closed_loop_pose(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> ClosedLoopTrajectoryPose | None:
        """
        Query MPS for ClosedLoopTrajectoryPose at a specific timestamp. This will throw an exception if open loop trajectory data is not available. Check for data availability first using `has_closed_loop_poses()`
        """
    def get_eyegaze_version(self) -> str | None:
        """
        Get the MPS eye gaze version.
        """
    def get_general_eyegaze(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> EyeGaze | None:
        """
        Query MPS for general EyeGaze at a specific timestamp. This will throw an exception if general eye gaze data is not available. Check for data availability first using: `has_general_eyegaze()`
        """
    def get_hand_tracking_version(self) -> str | None:
        """
        Get the MPS hand tracking version.
        """
    def get_online_calibration(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> OnlineCalibration | None:
        """
        Query MPS for OnlineCalibration at a specific timestamp. This will throw an exception if online calibration data is not available. Check for data availability first using `has_online_calibrations()`
        """
    def get_open_loop_pose(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> OpenLoopTrajectoryPose | None:
        """
        Query MPS for OpenLoopTrajectoryPose at a specific timestamp. This will throw an exception if open loop trajectory data is not available. Check for data availability first using `has_open_loop_poses()`
        """
    def get_personalized_eyegaze(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> EyeGaze | None:
        """
        Query MPS for personalized EyeGaze at a specific timestamp. This will throw an exception if personalized eye gaze data is not available. Check for data availability first using `has_personalized_eyegaze()`
        """
    def get_rgb_corrected_closed_loop_pose(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> SE3 | None:
        """
        Get the corrected rgb frame pose based on the online calibration.
        """
    def get_rgb_corrected_timestamp_ns(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> int | None:
        """
        Get the corrected rgb frame timestamp based on the online calibration.
        """
    def get_semidense_observations(self) -> list[PointObservation]:
        """
        Get the MPS point observations. This will throw an exception if the observations are not available. Check for data availability first using 'has_semidense_observations()'
        """
    def get_semidense_point_cloud(self) -> list[GlobalPointPosition]:
        """
        Get the MPS semidense point cloud. This will throw an exception if the point cloud is not available. Check for data availability first using 'has_semidense_point_cloud()'
        """
    def get_slam_version(self) -> str | None:
        """
        Get the MPS SLAM version.
        """
    def get_wrist_and_palm_pose(self, arg0: int, arg1: projectaria_tools.core.sensor_data.TimeQueryOptions) -> ... | None:
        """
        Get the MPS wrist and palm pose. This will throw an exception if the wrist and palm poses are not available. Check for data availability first using 'has_wrist_and_palm_poses()'
        """
    def has_closed_loop_poses(self) -> bool:
        """
        Check if closed loop poses are available in the MPS data paths
        """
    def has_general_eyegaze(self) -> bool:
        """
        Check if general eye gaze data is available in the MPS data paths
        """
    def has_online_calibrations(self) -> bool:
        """
        Check if online calibrations are available in the MPS data paths
        """
    def has_open_loop_poses(self) -> bool:
        """
        Check if open loop poses are available in the MPS data paths
        """
    def has_personalized_eyegaze(self) -> bool:
        """
        Check if personalized eye gaze data is available in the MPS data paths
        """
    def has_semidense_observations(self) -> bool:
        """
        Check if semidense observations are available in the MPS data paths
        """
    def has_semidense_point_cloud(self) -> bool:
        """
        Check if semidense point cloud data is available in the MPS data paths
        """
    def has_wrist_and_palm_poses(self) -> bool:
        """
        Check if wrist and palm poses are available in the MPS data paths
        """
class MpsEyegazeDataPaths:
    """
    A struct that includes the file paths of all MPS eye gaze data for a sequence.
    """
    general_eyegaze: str
    personalized_eyegaze: str
    summary: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class MpsSlamDataPaths:
    """
    A struct that includes the file paths of all MPS SLAM data for a VRS sequence processed by MPS.
    """
    closed_loop_trajectory: str
    online_calibrations: str
    open_loop_trajectory: str
    semidense_observations: str
    semidense_points: str
    summary: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class OnlineCalibration:
    camera_calibs: list[projectaria_tools.core.calibration.CameraCalibration]
    imu_calibs: list[projectaria_tools.core.calibration.ImuCalibration]
    tracking_timestamp: datetime.timedelta
    utc_timestamp: datetime.timedelta
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
    def get_camera_calib(self, label: str) -> projectaria_tools.core.calibration.CameraCalibration | None:
        """
        Helper function to get the camera calibration of a specific camera label
        """
class OpenLoopTrajectoryPose:
    """
    
            Open loop trajectory is the odometry estimation output by the visual-inertial odometry (VIO), in
            an arbitrary odometry coordinate frame. The estimation includes pose and dynamics (translational
            and angular velocities).
    
            The open loop trajectory has good “relative” and “local” accuracy: the relative transformation
            between two frames is accurate when the time span between two frames is short (within a few
            minutes). However, the open loop trajectory has increased drift error accumulated over time spent
            and travel distance. Consider using closed loop trajectory if you are looking for trajectory
            without drift error.
        
    """
    angular_velocity_device: numpy.ndarray[numpy.float64[3, 1]]
    device_linear_velocity_odometry: numpy.ndarray[numpy.float64[3, 1]]
    gravity_odometry: numpy.ndarray[numpy.float64[3, 1]]
    quality_score: float
    session_uid: str
    tracking_timestamp: datetime.timedelta
    transform_odometry_device: SE3
    utc_timestamp: datetime.timedelta
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class PointObservation:
    """
    2D observations of the point
    """
    camera_serial: str
    frame_capture_timestamp: datetime.timedelta
    point_uid: int
    uv: numpy.ndarray[numpy.float32[2, 1]]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class StaticCameraCalibration:
    """
    Static camera intrinsic calibration and extrinsics in the world frame
    """
    camera_uid: str
    end_frame_idx: int | None
    graph_uid: str
    height: int
    intrinsics: numpy.ndarray[numpy.float32[8, 1]]
    intrinsics_type: str
    quality: int
    start_frame_idx: int | None
    transform_world_cam: SE3
    width: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class StreamCompressionMode:
    """
    Stream compression mode
    
    Members:
    
      NONE : No compression
    
      GZIP : GZIP compression
    """
    GZIP: typing.ClassVar[StreamCompressionMode]  # value = <StreamCompressionMode.GZIP: 1>
    NONE: typing.ClassVar[StreamCompressionMode]  # value = <StreamCompressionMode.NONE: 0>
    __members__: typing.ClassVar[typing.Dict[str, StreamCompressionMode]]  # value = {'NONE': <StreamCompressionMode.NONE: 0>, 'GZIP': <StreamCompressionMode.GZIP: 1>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: object) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: object) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(arg0: StreamCompressionMode) -> int:
        ...
def compute_depth_and_combined_gaze_direction(left_yaw_rads: float, right_yaw_rads: float, pitch_rads: float) -> tuple[float, float, float]:
    """
     Given the left and right yaw angles and common pitch get the combined gaze angles and depth in CPF frame.
      Parameters
      __________
      left_yaw_rads: Left Yaw angle in radians in CPF frame.
      right_yaw_rads: Right Yaw angle in radians in CPF frame.
      pitch_rads: Pitch angle in radians in CPF frame.
    """
def get_eyegaze_point_at_depth(yaw_rads: float, pitch_rads: float, depth_m: float) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
     Given the yaw and pitch angles of the eye gaze and a depth, return the gaze 3D point in CPF frame.
      Parameters
      __________
      yaw_rads: Yaw angle in radians in CPF frame.
      pitch_rads: Pitch angle in radians in CPF frame.
      depth_m: Depth of the point in meters.
    """
def get_gaze_intersection_point(left_yaw_rads: float, right_yaw_rads: float, pitch_rads: float) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
     Given the left and right yaw angles and common pitch get the intersection point in 3D in CPF frame.
      Parameters
      __________
      left_yaw_rads: Left Yaw angle in radians in CPF frame.
      right_yaw_rads: Right Yaw angle in radians in CPF frame.
      pitch_rads: Pitch angle in radians in CPF frame.
    """
def get_gaze_vectors(left_yaw_rads: float, right_yaw_rads: float, pitch_rads: float) -> tuple[numpy.ndarray[numpy.float64[3, 1]], numpy.ndarray[numpy.float64[3, 1]]]:
    """
     Given the left and right yaw angles and common pitch get the left and right gaze vectors from their respective origins in XYZ CPF frame.
      Parameters
      __________
      left_yaw_rads: Left Yaw angle in radians in CPF frame.
      right_yaw_rads: Right Yaw angle in radians in CPF frame.
      pitch_rads: Pitch angle in radians in CPF frame.
    """
def read_closed_loop_trajectory(path: str) -> list[ClosedLoopTrajectoryPose]:
    """
    Read Closed loop trajectory.
      Parameters
      __________
      path: Path to the closed loop trajectory csv file. Usually named 'closed_loop_trajectory.csv'
    """
def read_eyegaze(path: str) -> list[EyeGaze]:
    """
    Read Eye Gaze from the eye gaze output generated via MPS.
      Parameters
      __________
      path: Path to the eye gaze csv file.
    """
@typing.overload
def read_global_point_cloud(path: str, compression: StreamCompressionMode) -> list[GlobalPointPosition]:
    """
    Read global point cloud.
      Parameters
      __________
      path: Path to the global point cloud file. Usually named 'global_pointcloud.csv.gz'
      compression: Stream compression mode for reading csv file.
    """
@typing.overload
def read_global_point_cloud(path: str) -> list[GlobalPointPosition]:
    """
    Read global point cloud.
      Parameters
      __________
      path: Path to the global point cloud file. Usually named 'global_pointcloud' with a '.csv' or '.csv.gz'
    """
def read_online_calibration(path: str) -> list[OnlineCalibration]:
    """
    Read estimated online calibrations.
      Parameters
      __________
      path: Path to the online calibration jsonl file. Usually named 'online_calibration.jsonl'
    """
def read_open_loop_trajectory(path: str) -> list[OpenLoopTrajectoryPose]:
    """
    Read Open loop trajectory.
      Parameters
      __________
      path: Path to the open loop trajectory csv file. Usually named 'open_loop_trajectory.csv'
    """
@typing.overload
def read_point_observations(path: str, compression: StreamCompressionMode) -> list[PointObservation]:
    """
    Read point observations.
      Parameters
      __________
      path: Path to the point observations file. Usually named 'semidense_observations.csv.gz'
      compression: Stream compression mode for reading csv file.
    """
@typing.overload
def read_point_observations(path: str) -> list[PointObservation]:
    """
    Read point observations.
      Parameters
      __________
      path: Path to the point observations file. Usually named 'semidense_observations' with a '.csv' or '.csv.gz'
    """
def read_static_camera_calibrations(path: str) -> list[StaticCameraCalibration]:
    """
    Read static camera calibrations.
      Parameters
      __________
      path: Path to the static camera calibrations file.
    """
