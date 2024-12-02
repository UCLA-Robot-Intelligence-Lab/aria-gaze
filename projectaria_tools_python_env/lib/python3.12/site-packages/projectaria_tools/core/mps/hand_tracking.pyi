from __future__ import annotations
import datetime
import typing
__all__ = ['WristAndPalmPose', 'read_wrist_and_palm_poses']
class WristAndPalmPose:
    """
    An object representing WristAndPalmPose output at a single timestamp.
    """
    left_hand: ... | None
    right_hand: ... | None
    tracking_timestamp: datetime.timedelta
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
def read_wrist_and_palm_poses(path: str) -> list[WristAndPalmPose]:
    """
    Read Wrist and Palm poses from the hand tracking output generated via MPS.
      Parameters
      __________
      path: Path to the wrist and palm poses csv file.
    """
