from __future__ import annotations
__all__ = ['Settings', 'run']
class Settings:
    """
    Vrs health check settings.
    """
    default_gps_rate_hz: float
    default_imu_period_us: float
    ignore_audio: bool
    ignore_bluetooth: bool
    ignore_gps: bool
    is_interactive: bool
    max_allowed_rotation_accel_rad_per_s2: float
    max_camera_exposure_ms: float
    max_camera_gain: float
    max_frame_drop_us: int
    max_imu_skip_us: float
    max_non_physical_accel: float
    max_temp: float
    min_alignment_score: float
    min_audio_score: float
    min_baro_score: float
    min_camera_exposure_ms: float
    min_camera_gain: float
    min_camera_score: float
    min_gps_accuracy: float
    min_imu_score: float
    min_temp: float
    min_time_domain_mapping_score: float
    physical_accel_threshold: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
def run(path: str, json_out_filename: str = ..., settings: Settings = ..., dropped_out_filename: str = ..., print_stats: bool = ..., disable_logging: bool = ...) -> int:
    """
    Run vrs health check.
    """
