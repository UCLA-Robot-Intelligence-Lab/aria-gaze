from __future__ import annotations
import projectaria_tools.core.stream_id
import numpy
import pybind11_stubgen.typing_ext
import typing
__all__ = ['AFTER', 'AUDIO', 'AudioConfig', 'AudioData', 'AudioDataRecord', 'BAROMETER', 'BEFORE', 'BLUETOOTH', 'BarometerConfigRecord', 'BarometerData', 'BluetoothBeaconConfigRecord', 'BluetoothBeaconData', 'CLOSEST', 'DEVICE_TIME', 'GPS', 'GpsConfigRecord', 'GpsData', 'HOST_TIME', 'IMAGE', 'IMU', 'ImageConfigRecord', 'ImageData', 'ImageDataRecord', 'MAGNETOMETER', 'MotionConfigRecord', 'MotionData', 'NOT_VALID', 'PixelFrame', 'RECORD_TIME', 'SensorConfiguration', 'SensorData', 'SensorDataType', 'TIC_SYNC', 'TIME_CODE', 'TimeDomain', 'TimeQueryOptions', 'TimeSyncMode', 'WPS', 'WifiBeaconConfigRecord', 'WifiBeaconData', 'get_sensor_data_type_name', 'get_time_domain_name', 'has_calibration', 'supports_host_time_domain']
class AudioConfig:
    """
    Audio sensor configuration type
    """
    num_channels: int
    sample_format: int
    sample_rate: int
    stream_id: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class AudioData:
    """
    Audio sensor data type: the audio value
    """
    data: list[int]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class AudioDataRecord:
    """
    Audio meta data
    """
    audio_muted: int
    capture_timestamps_ns: list[int]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class BarometerConfigRecord:
    """
    Barometer sensor configuration type
    """
    sample_rate: float
    sensor_model_name: str
    stream_id: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class BarometerData:
    capture_timestamp_ns: int
    pressure: float
    temperature: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class BluetoothBeaconConfigRecord:
    """
    Bluetooth sensor configuration type
    """
    sample_rate_hz: float
    streamId: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class BluetoothBeaconData:
    board_scan_request_complete_timestamp_ns: int
    board_scan_request_start_timestamp_ns: int
    board_timestamp_ns: int
    freq_mhz: float
    rssi: float
    system_timestamp_ns: int
    tx_power: float
    unique_id: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class GpsConfigRecord:
    """
    Gps sensor configuration type
    """
    sample_rate_hz: float
    stream_id: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class GpsData:
    """
    Gps data type, note that GPS sensor data are already rectified
    """
    accuracy: float
    altitude: float
    capture_timestamp_ns: int
    latitude: float
    longitude: float
    provider: str
    raw_data: list[str]
    speed: float
    utc_time_ms: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class ImageConfigRecord:
    camera_id: int
    description: str
    device_serial: str
    device_type: str
    device_version: str
    exposure_duration_max: float
    exposure_duration_min: float
    factory_calibration: str
    gain_max: float
    gain_min: float
    gamma_factor: float
    image_height: int
    image_stride: int
    image_width: int
    nominal_rate_hz: float
    online_calibration: str
    pixel_format: int
    sensor_model: str
    sensor_serial: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class ImageData:
    pixel_frame: PixelFrame
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns pixel value at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns number of rows in image
        """
    def get_width(self) -> int:
        """
        Returns number of columns in image
        """
    def is_valid(self) -> bool:
        """
        Returns if image is empty
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ImageDataRecord:
    arrival_timestamp_ns: int
    camera_id: int
    capture_timestamp_ns: int
    exposure_duration: float
    frame_number: int
    gain: float
    group_id: int
    group_mask: int
    temperature: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class MotionConfigRecord:
    description: str
    device_id: int
    device_serial: str
    device_type: str
    factory_calibration: str
    has_accelerometer: bool
    has_gyroscope: bool
    has_magnetometer: bool
    nominal_rate_hz: float
    online_calibration: str
    sensor_model: str
    stream_index: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class MotionData:
    accel_msec2: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]
    accel_valid: bool
    arrival_timestamp_ns: int
    capture_timestamp_ns: int
    gyro_radsec: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]
    gyro_valid: bool
    mag_tesla: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]
    mag_valid: bool
    temperature: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class PixelFrame:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def get_buffer(self) -> list[int]:
        """
        Get image data buffer
        """
    def get_height(self) -> int:
        """
        Return number of rows in image
        """
    def get_width(self) -> int:
        """
        Return number of columns in image
        """
    def normalize_frame(self, arg0: bool) -> PixelFrame:
        """
        Normalize an input frame if possible and as necessary
        """
class SensorConfiguration:
    """
    Configuration of a sensor stream, such as stream id, nominal frame rate
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: None | ImageConfigRecord | MotionConfigRecord | GpsConfigRecord | WifiBeaconConfigRecord | AudioConfig | BarometerConfigRecord | BluetoothBeaconConfigRecord, arg1: SensorDataType) -> None:
        ...
    def audio_configuration(self) -> AudioConfig:
        """
        Returns the sensor configuration as AudioConfig
        """
    def barometer_configuration(self) -> BarometerConfigRecord:
        """
        Returns the sensor configuration as BarometerConfigRecord
        """
    def bluetooth_configuration(self) -> BluetoothBeaconConfigRecord:
        """
        Returns the sensor configuration as Bluetooth
        """
    def get_nominal_rate_hz(self) -> float:
        """
        Returns the nominal frame rate of the sensor
        """
    def gps_configuration(self) -> GpsConfigRecord:
        """
        Returns the sensor configuration as GpsConfigRecord
        """
    def image_configuration(self) -> ImageConfigRecord:
        """
        Returns the sensor configuration as ImageConfigRecord
        """
    def magnetometer_configuration(self) -> MotionConfigRecord:
        """
        Returns the sensor configuration as MotionConfigRecord
        """
    def motion_configuration(self) -> MotionConfigRecord:
        """
        Returns the sensor configuration as MotionConfigRecord
        """
    def sensor_data_type(self) -> SensorDataType:
        """
        Returns the type of sensor data 
        """
    def wps_configuration(self) -> WifiBeaconConfigRecord:
        """
        Returns the sensor configuration as WifiBeaconConfigRecord
        """
class SensorData:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: projectaria_tools.core.stream_id.StreamId, arg1: None | tuple[ImageData, ImageDataRecord] | MotionData | GpsData | WifiBeaconData | tuple[AudioData, AudioDataRecord] | BarometerData | BluetoothBeaconData, arg2: SensorDataType, arg3: int, arg4: dict[TimeSyncMode, int]) -> None:
        ...
    def audio_data_and_record(self) -> tuple[AudioData, AudioDataRecord]:
        ...
    def barometer_data(self) -> BarometerData:
        ...
    def bluetooth_data(self) -> BluetoothBeaconData:
        ...
    def get_time_ns(self, time_domain: TimeDomain) -> int:
        ...
    def gps_data(self) -> GpsData:
        ...
    def image_data_and_record(self) -> tuple[ImageData, ImageDataRecord]:
        ...
    def imu_data(self) -> MotionData:
        ...
    def magnetometer_data(self) -> MotionData:
        ...
    def sensor_data_type(self) -> SensorDataType:
        ...
    def stream_id(self) -> projectaria_tools.core.stream_id.StreamId:
        ...
    def wps_data(self) -> WifiBeaconData:
        ...
class SensorDataType:
    """
    Enum class for different types of sensor data used in projectaria_tools
    
    Members:
    
      NOT_VALID
    
      IMAGE : camera image streams
    
      IMU : Inertial measurement unit (IMU) data streams, including accelerometer and gyroscope, note that magnetometer is a different stream
    
      GPS : Global positioning system (GPS) data streams
    
      WPS : Wifi beacon data streams
    
      AUDIO : Audio data streams
    
      BAROMETER : Barometer data streams
    
      BLUETOOTH : Bluetooth data streams
    
      MAGNETOMETER : Magnetometer data streams
    """
    AUDIO: typing.ClassVar[SensorDataType]  # value = <SensorDataType.AUDIO: 5>
    BAROMETER: typing.ClassVar[SensorDataType]  # value = <SensorDataType.BAROMETER: 6>
    BLUETOOTH: typing.ClassVar[SensorDataType]  # value = <SensorDataType.BLUETOOTH: 7>
    GPS: typing.ClassVar[SensorDataType]  # value = <SensorDataType.GPS: 3>
    IMAGE: typing.ClassVar[SensorDataType]  # value = <SensorDataType.IMAGE: 1>
    IMU: typing.ClassVar[SensorDataType]  # value = <SensorDataType.IMU: 2>
    MAGNETOMETER: typing.ClassVar[SensorDataType]  # value = <SensorDataType.MAGNETOMETER: 8>
    NOT_VALID: typing.ClassVar[SensorDataType]  # value = <SensorDataType.NOT_VALID: 0>
    WPS: typing.ClassVar[SensorDataType]  # value = <SensorDataType.WPS: 4>
    __members__: typing.ClassVar[typing.Dict[str, SensorDataType]]  # value = {'NOT_VALID': <SensorDataType.NOT_VALID: 0>, 'IMAGE': <SensorDataType.IMAGE: 1>, 'IMU': <SensorDataType.IMU: 2>, 'GPS': <SensorDataType.GPS: 3>, 'WPS': <SensorDataType.WPS: 4>, 'AUDIO': <SensorDataType.AUDIO: 5>, 'BAROMETER': <SensorDataType.BAROMETER: 6>, 'BLUETOOTH': <SensorDataType.BLUETOOTH: 7>, 'MAGNETOMETER': <SensorDataType.MAGNETOMETER: 8>}
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
    def value(arg0: SensorDataType) -> int:
        ...
class TimeDomain:
    """
    Enum class for different types of timestamps used in projectaria_tools
    
    Members:
    
      RECORD_TIME : timestamp directly stored in vrs index, fast to access, but not guaranteed which time domain
    
      DEVICE_TIME : capture time in device's timedomain, <b>accurate</b>. All sensors on the same Aria glass share the same device time domain as they are issued from the same clock. We <b>strongly recommend</b> to always work with the device timestamp when dealing with <b>single-device</b> Aria data.
    
      HOST_TIME : arrival time in host computer's timedomain, may not be accurate
    
      TIME_CODE : capture in TimeSync server's timedomain, accurate across devices in a <b>multi-device</b> data capture.
    
      TIC_SYNC : capture in TimeSync server's timedomain where the server can be an Aria device, accurate across devices in a <b>multi-device</b> data capture
    """
    DEVICE_TIME: typing.ClassVar[TimeDomain]  # value = <TimeDomain.DEVICE_TIME: 1>
    HOST_TIME: typing.ClassVar[TimeDomain]  # value = <TimeDomain.HOST_TIME: 2>
    RECORD_TIME: typing.ClassVar[TimeDomain]  # value = <TimeDomain.RECORD_TIME: 0>
    TIC_SYNC: typing.ClassVar[TimeDomain]  # value = <TimeDomain.TIC_SYNC: 4>
    TIME_CODE: typing.ClassVar[TimeDomain]  # value = <TimeDomain.TIME_CODE: 3>
    __members__: typing.ClassVar[typing.Dict[str, TimeDomain]]  # value = {'RECORD_TIME': <TimeDomain.RECORD_TIME: 0>, 'DEVICE_TIME': <TimeDomain.DEVICE_TIME: 1>, 'HOST_TIME': <TimeDomain.HOST_TIME: 2>, 'TIME_CODE': <TimeDomain.TIME_CODE: 3>, 'TIC_SYNC': <TimeDomain.TIC_SYNC: 4>}
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
    def value(arg0: TimeDomain) -> int:
        ...
class TimeQueryOptions:
    """
    Members:
    
      BEFORE : the last valid data with `timestamp <= t_query
    
      AFTER : the first valid data with `timestamp >= t_query
    
      CLOSEST : the data whose `|timestamp - t_query|` is smallest
    """
    AFTER: typing.ClassVar[TimeQueryOptions]  # value = <TimeQueryOptions.AFTER: 1>
    BEFORE: typing.ClassVar[TimeQueryOptions]  # value = <TimeQueryOptions.BEFORE: 0>
    CLOSEST: typing.ClassVar[TimeQueryOptions]  # value = <TimeQueryOptions.CLOSEST: 2>
    __members__: typing.ClassVar[typing.Dict[str, TimeQueryOptions]]  # value = {'BEFORE': <TimeQueryOptions.BEFORE: 0>, 'AFTER': <TimeQueryOptions.AFTER: 1>, 'CLOSEST': <TimeQueryOptions.CLOSEST: 2>}
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
    def value(arg0: TimeQueryOptions) -> int:
        ...
class TimeSyncMode:
    """
    Members:
    
      TIME_CODE : TIMECODE mode
    
      TIC_SYNC : TIC_SYNC mode
    """
    TIC_SYNC: typing.ClassVar[TimeSyncMode]  # value = <TimeSyncMode.TIC_SYNC: 1>
    TIME_CODE: typing.ClassVar[TimeSyncMode]  # value = <TimeSyncMode.TIME_CODE: 0>
    __members__: typing.ClassVar[typing.Dict[str, TimeSyncMode]]  # value = {'TIME_CODE': <TimeSyncMode.TIME_CODE: 0>, 'TIC_SYNC': <TimeSyncMode.TIC_SYNC: 1>}
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
    def value(arg0: TimeSyncMode) -> int:
        ...
class WifiBeaconConfigRecord:
    stream_id: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class WifiBeaconData:
    board_scan_request_complete_timestamp_ns: int
    board_scan_request_start_timestamp_ns: int
    board_timestamp_ns: int
    bssid_mac: str
    freq_mhz: float
    rssi: float
    rssi_per_antenna: list[float]
    ssid: str
    system_timestamp_ns: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
def get_sensor_data_type_name(arg0: SensorDataType) -> str:
    """
    converts the enum to readable string
    """
def get_time_domain_name(arg0: TimeDomain) -> str:
    """
    A helper function to return a descriptive name for a given TimeDomain enum
    """
def has_calibration(type: SensorDataType) -> bool:
    """
    checks if calibration exists for a specific stream
    """
def supports_host_time_domain(type: SensorDataType) -> bool:
    """
    checks if host time domain is supported by a type. Note we encourage user to avoid using host time domains as arrival timestamps are inaccurate.
    """
AFTER: TimeQueryOptions  # value = <TimeQueryOptions.AFTER: 1>
AUDIO: SensorDataType  # value = <SensorDataType.AUDIO: 5>
BAROMETER: SensorDataType  # value = <SensorDataType.BAROMETER: 6>
BEFORE: TimeQueryOptions  # value = <TimeQueryOptions.BEFORE: 0>
BLUETOOTH: SensorDataType  # value = <SensorDataType.BLUETOOTH: 7>
CLOSEST: TimeQueryOptions  # value = <TimeQueryOptions.CLOSEST: 2>
DEVICE_TIME: TimeDomain  # value = <TimeDomain.DEVICE_TIME: 1>
GPS: SensorDataType  # value = <SensorDataType.GPS: 3>
HOST_TIME: TimeDomain  # value = <TimeDomain.HOST_TIME: 2>
IMAGE: SensorDataType  # value = <SensorDataType.IMAGE: 1>
IMU: SensorDataType  # value = <SensorDataType.IMU: 2>
MAGNETOMETER: SensorDataType  # value = <SensorDataType.MAGNETOMETER: 8>
NOT_VALID: SensorDataType  # value = <SensorDataType.NOT_VALID: 0>
RECORD_TIME: TimeDomain  # value = <TimeDomain.RECORD_TIME: 0>
TIC_SYNC: TimeSyncMode  # value = <TimeSyncMode.TIC_SYNC: 1>
TIME_CODE: TimeSyncMode  # value = <TimeSyncMode.TIME_CODE: 0>
WPS: SensorDataType  # value = <SensorDataType.WPS: 4>
