from __future__ import annotations
import projectaria_tools.core.calibration
from projectaria_tools.core.calibration import CameraCalibration
from projectaria_tools.core.calibration import DeviceCalibration
import projectaria_tools.core.mps
from projectaria_tools.core.mps import ClosedLoopTrajectoryPose
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps import GlobalPointPosition
from projectaria_tools.core.mps import get_eyegaze_point_at_depth
import projectaria_tools.core.mps.hand_tracking
from projectaria_tools.core.mps import hand_tracking
from projectaria_tools.core.sophus import SE3
import numpy as np
import numpy
import typing
__all__ = ['CameraCalibration', 'ClosedLoopTrajectoryPose', 'DeviceCalibration', 'EyeGaze', 'GlobalPointPosition', 'List', 'SE3', 'bisection_timestamp_search', 'filter_points_from_confidence', 'filter_points_from_count', 'get_eyegaze_point_at_depth', 'get_gaze_vector_reprojection', 'get_nearest_eye_gaze', 'get_nearest_pose', 'get_nearest_wrist_and_palm_pose', 'hand_tracking', 'np']
def bisection_timestamp_search(timed_data, query_timestamp_ns: int) -> int:
    """
    
        Binary search helper function, assuming that timed_data is sorted by the field names 'tracking_timestamp'
        Returns index of the element closest to the query timestamp else returns None if not found (out of time range)
        
    """
def filter_points_from_confidence(raw_points: typing.List[projectaria_tools.core.mps.GlobalPointPosition], threshold_invdep: float = 0.001, threshold_dep: float = 0.05) -> typing.List[projectaria_tools.core.mps.GlobalPointPosition]:
    """
    
        Filter the point cloud by inv depth and depth
        
    """
def filter_points_from_count(raw_points: typing.List[projectaria_tools.core.mps.GlobalPointPosition], max_point_count: int = 500000) -> typing.List[projectaria_tools.core.mps.GlobalPointPosition]:
    """
    
        Filter the point cloud by count (random points are sampled from the initial set)
        
    """
def get_gaze_vector_reprojection(eye_gaze: projectaria_tools.core.mps.EyeGaze, stream_id_label: str, device_calibration: projectaria_tools.core.calibration.DeviceCalibration, camera_calibration: projectaria_tools.core.calibration.CameraCalibration, depth_m: float = 1.0, make_upright: bool = ...) -> numpy.ndarray:
    """
    
        Helper function to project a eye gaze output onto a given image and its calibration, assuming specified fixed depth
        
    """
def get_nearest_eye_gaze(eye_gazes: typing.List[projectaria_tools.core.mps.EyeGaze], query_timestamp_ns: int) -> projectaria_tools.core.mps.EyeGaze:
    """
    
        Helper function to get nearest eye gaze for a timestamp (ns)
        Return the closest or equal timestamp eye_gaze information that can be found, returns None if not found (out of time range)
        
    """
def get_nearest_pose(mps_trajectory: typing.List[projectaria_tools.core.mps.ClosedLoopTrajectoryPose], query_timestamp_ns: int) -> projectaria_tools.core.mps.ClosedLoopTrajectoryPose:
    """
    
        Helper function to get nearest pose for a timestamp (ns)
        Return the closest or equal timestamp pose information that can be found, returns None if not found (out of time range)
        
    """
def get_nearest_wrist_and_palm_pose(wirst_and_palm_poses: typing.List[projectaria_tools.core.mps.hand_tracking.WristAndPalmPose], query_timestamp_ns: int) -> projectaria_tools.core.mps.hand_tracking.WristAndPalmPose:
    """
    
        Helper function to get nearest wrist and palm pose for a timestamp (ns)
        Return the closest or equal timestamp wrist and palm pose that can be found, returns None if not found (out of time range)
        
    """
List: typing._SpecialGenericAlias  # value = typing.List
