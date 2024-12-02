from __future__ import annotations
import typing
__all__ = ['BAROMETER_RECORDABLE_CLASS', 'BLUETOOTH_BEACON_RECORDABLE_CLASS', 'EYE_CAMERA_RECORDABLE_CLASS', 'GPS_RECORDABLE_CLASS', 'IMU_RECORDABLE_CLASS', 'RGB_CAMERA_RECORDABLE_CLASS', 'RecordableTypeId', 'SLAM_CAMERA_DATA', 'SLAM_IMU_DATA', 'SLAM_MAGNETOMETER_DATA', 'STEREO_AUDIO_RECORDABLE_CLASS', 'StreamId', 'TIME_RECORDABLE_CLASS', 'WIFI_BEACON_RECORDABLE_CLASS']
class RecordableTypeId:
    """
    Recordable Type Id, e.g. SLAM_CAMERA_DATA
    
    Members:
    
      SLAM_CAMERA_DATA
    
      EYE_CAMERA_RECORDABLE_CLASS
    
      RGB_CAMERA_RECORDABLE_CLASS
    
      SLAM_IMU_DATA
    
      IMU_RECORDABLE_CLASS
    
      SLAM_MAGNETOMETER_DATA
    
      BAROMETER_RECORDABLE_CLASS
    
      GPS_RECORDABLE_CLASS
    
      WIFI_BEACON_RECORDABLE_CLASS
    
      BLUETOOTH_BEACON_RECORDABLE_CLASS
    
      STEREO_AUDIO_RECORDABLE_CLASS
    
      TIME_RECORDABLE_CLASS
    """
    BAROMETER_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.BAROMETER_RECORDABLE_CLASS: 247>
    BLUETOOTH_BEACON_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.BLUETOOTH_BEACON_RECORDABLE_CLASS: 283>
    EYE_CAMERA_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.EYE_CAMERA_RECORDABLE_CLASS: 211>
    GPS_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.GPS_RECORDABLE_CLASS: 281>
    IMU_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.IMU_RECORDABLE_CLASS: 241>
    RGB_CAMERA_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS: 214>
    SLAM_CAMERA_DATA: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.SLAM_CAMERA_DATA: 1201>
    SLAM_IMU_DATA: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.SLAM_IMU_DATA: 1202>
    SLAM_MAGNETOMETER_DATA: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.SLAM_MAGNETOMETER_DATA: 1203>
    STEREO_AUDIO_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.STEREO_AUDIO_RECORDABLE_CLASS: 231>
    TIME_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.TIME_RECORDABLE_CLASS: 285>
    WIFI_BEACON_RECORDABLE_CLASS: typing.ClassVar[RecordableTypeId]  # value = <RecordableTypeId.WIFI_BEACON_RECORDABLE_CLASS: 282>
    __members__: typing.ClassVar[typing.Dict[str, RecordableTypeId]]  # value = {'SLAM_CAMERA_DATA': <RecordableTypeId.SLAM_CAMERA_DATA: 1201>, 'EYE_CAMERA_RECORDABLE_CLASS': <RecordableTypeId.EYE_CAMERA_RECORDABLE_CLASS: 211>, 'RGB_CAMERA_RECORDABLE_CLASS': <RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS: 214>, 'SLAM_IMU_DATA': <RecordableTypeId.SLAM_IMU_DATA: 1202>, 'IMU_RECORDABLE_CLASS': <RecordableTypeId.IMU_RECORDABLE_CLASS: 241>, 'SLAM_MAGNETOMETER_DATA': <RecordableTypeId.SLAM_MAGNETOMETER_DATA: 1203>, 'BAROMETER_RECORDABLE_CLASS': <RecordableTypeId.BAROMETER_RECORDABLE_CLASS: 247>, 'GPS_RECORDABLE_CLASS': <RecordableTypeId.GPS_RECORDABLE_CLASS: 281>, 'WIFI_BEACON_RECORDABLE_CLASS': <RecordableTypeId.WIFI_BEACON_RECORDABLE_CLASS: 282>, 'BLUETOOTH_BEACON_RECORDABLE_CLASS': <RecordableTypeId.BLUETOOTH_BEACON_RECORDABLE_CLASS: 283>, 'STEREO_AUDIO_RECORDABLE_CLASS': <RecordableTypeId.STEREO_AUDIO_RECORDABLE_CLASS: 231>, 'TIME_RECORDABLE_CLASS': <RecordableTypeId.TIME_RECORDABLE_CLASS: 285>}
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
    def value(arg0: RecordableTypeId) -> int:
        ...
class StreamId:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, arg0: StreamId) -> bool:
        """
        Compares two StreamIds
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_instance_id(self) -> int:
        """
        Returns the instance id of the Stream, range in 1...N
        """
    def get_name(self) -> str:
        """
        Returns the name of the StreamId
        """
    def get_type_id(self) -> RecordableTypeId:
        """
        Returns the RecordableTypeId of the StreamId
        """
    def get_type_name(self) -> str:
        """
        Returns the type name of the StreamId
        """
    def is_valid(self) -> bool:
        """
        Returns if a stream is valid
        """
BAROMETER_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.BAROMETER_RECORDABLE_CLASS: 247>
BLUETOOTH_BEACON_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.BLUETOOTH_BEACON_RECORDABLE_CLASS: 283>
EYE_CAMERA_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.EYE_CAMERA_RECORDABLE_CLASS: 211>
GPS_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.GPS_RECORDABLE_CLASS: 281>
IMU_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.IMU_RECORDABLE_CLASS: 241>
RGB_CAMERA_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS: 214>
SLAM_CAMERA_DATA: RecordableTypeId  # value = <RecordableTypeId.SLAM_CAMERA_DATA: 1201>
SLAM_IMU_DATA: RecordableTypeId  # value = <RecordableTypeId.SLAM_IMU_DATA: 1202>
SLAM_MAGNETOMETER_DATA: RecordableTypeId  # value = <RecordableTypeId.SLAM_MAGNETOMETER_DATA: 1203>
STEREO_AUDIO_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.STEREO_AUDIO_RECORDABLE_CLASS: 231>
TIME_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.TIME_RECORDABLE_CLASS: 285>
WIFI_BEACON_RECORDABLE_CLASS: RecordableTypeId  # value = <RecordableTypeId.WIFI_BEACON_RECORDABLE_CLASS: 282>
