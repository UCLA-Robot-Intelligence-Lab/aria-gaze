from __future__ import annotations
import projectaria_tools.core.calibration
import projectaria_tools.core.sensor_data
import projectaria_tools.core.stream_id
import typing
__all__ = ['DeliverQueuedOptions', 'MetadataTimeSyncMode', 'SensorDataIterator', 'SensorDataSequence', 'SubstreamSelector', 'VrsDataProvider', 'VrsMetadata', 'create_vrs_data_provider']
class DeliverQueuedOptions(SubstreamSelector):
    """
    Options for delivering sensor data of multiple streams sorted in device timestamps.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: int, arg1: int, arg2: dict[projectaria_tools.core.stream_id.StreamId, int]) -> None:
        ...
    def get_subsample_rate(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> int:
        """
        Returns how many times the frame rate is downsampled in a stream.
        """
    def get_truncate_first_device_time_ns(self) -> int:
        """
        Returns how many nanoseconds to skip from the beginning of the vrs recording.
        """
    def get_truncate_last_device_time_ns(self) -> int:
        """
        Returns how many nanoseconds to skip before the end of the vrs recording.
        """
    def set_subsample_rate(self, stream_id: projectaria_tools.core.stream_id.StreamId, rate: int) -> None:
        """
        Sets how many times the frame rate is downsampled in a stream i.e, after a data is played, rate - 1 data are skipped.
        """
    def set_truncate_first_device_time_ns(self, time_ns: int) -> None:
        """
        Sets how much time to skip from the beginning of the recording.
        """
    def set_truncate_last_device_time_ns(self, time_ns: int) -> None:
        """
        Sets how much time to skip from the end of the recording.
        """
class MetadataTimeSyncMode:
    """
    Members:
    
      NotEnabled
    
      Timecode
    
      Ntp
    
      TicSyncClient
    
      TicSyncServer
    """
    NotEnabled: typing.ClassVar[MetadataTimeSyncMode]  # value = <MetadataTimeSyncMode.NotEnabled: 0>
    Ntp: typing.ClassVar[MetadataTimeSyncMode]  # value = <MetadataTimeSyncMode.Ntp: 2>
    TicSyncClient: typing.ClassVar[MetadataTimeSyncMode]  # value = <MetadataTimeSyncMode.TicSyncClient: 3>
    TicSyncServer: typing.ClassVar[MetadataTimeSyncMode]  # value = <MetadataTimeSyncMode.TicSyncServer: 4>
    Timecode: typing.ClassVar[MetadataTimeSyncMode]  # value = <MetadataTimeSyncMode.Timecode: 1>
    __members__: typing.ClassVar[typing.Dict[str, MetadataTimeSyncMode]]  # value = {'NotEnabled': <MetadataTimeSyncMode.NotEnabled: 0>, 'Timecode': <MetadataTimeSyncMode.Timecode: 1>, 'Ntp': <MetadataTimeSyncMode.Ntp: 2>, 'TicSyncClient': <MetadataTimeSyncMode.TicSyncClient: 3>, 'TicSyncServer': <MetadataTimeSyncMode.TicSyncServer: 4>}
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
    def value(arg0: MetadataTimeSyncMode) -> int:
        ...
class SensorDataIterator:
    """
    Forward iterator for a sensor data container
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class SensorDataSequence:
    """
    Interface for delivering sensor data sorted by timestamps, with iterator support.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: ..., arg1: DeliverQueuedOptions) -> None:
        ...
class SubstreamSelector:
    """
    Class for subselecting VRS streams from all streams available in an VRS file.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: set[projectaria_tools.core.stream_id.StreamId]) -> None:
        ...
    @typing.overload
    def activate_stream(self, arg0: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Activate a VRS stream (turn on).
        """
    @typing.overload
    def activate_stream(self, arg0: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Turns on all streams of a specific typeId, regardless of current state.
        """
    def activate_stream_all(self) -> None:
        """
        Turns on all available streams, regardless of current state.
        """
    @typing.overload
    def deactivate_stream(self, arg0: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Deactivate a VRS stream (turn off).
        """
    @typing.overload
    def deactivate_stream(self, arg0: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Turns on all streams of a specific typeId, regardless of current state.
        """
    def deactivate_stream_all(self) -> None:
        """
        Turns off all available streams, regardless of current state.
        """
    def get_active_stream_ids(self) -> list[projectaria_tools.core.stream_id.StreamId]:
        """
        Returns all selected streams.
        """
    @typing.overload
    def get_stream_ids(self) -> list[projectaria_tools.core.stream_id.StreamId]:
        """
        Returns the list of available stream ids.
        """
    @typing.overload
    def get_stream_ids(self, arg0: projectaria_tools.core.stream_id.RecordableTypeId) -> list[projectaria_tools.core.stream_id.StreamId]:
        """
        Returns the list of stream ids of a specified type.
        """
    def get_type_ids(self) -> list[projectaria_tools.core.stream_id.RecordableTypeId]:
        """
        Returns the list of available type ids.
        """
    def is_active(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Returns true if a stream has been selected.
        """
    def toggle_stream(self, arg0: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Toggles a VRS stream from on to off or from off to on.
        """
class VrsDataProvider:
    """
    Given a vrs file that contains data collected from Aria devices, createVrsDataProvider will create and return a new VrsDataProvider object. A VrsDataProvider object can be used to access sensor data from a vrs file including image data, IMU data, calibration data and more.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: ..., arg1: ..., arg2: ..., arg3: ..., arg4: projectaria_tools.core.calibration.DeviceCalibration | None) -> None:
        ...
    def check_stream_is_active(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> bool:
        """
        Check, if a stream with provided ID is active.
        """
    def check_stream_is_type(self, stream_id: projectaria_tools.core.stream_id.StreamId, type: projectaria_tools.core.sensor_data.SensorDataType) -> bool:
        """
        Checks, if a stream with provided ID is of expected type.
        """
    def convert_from_device_time_to_synctime_ns(self, device_time_ns: int, mode: projectaria_tools.core.sensor_data.TimeSyncMode) -> int:
        """
        Convert DeviceTime timestamp into synchronized timestamp in nanoseconds.
        """
    def convert_from_device_time_to_timecode_ns(self, device_time_ns: int) -> int:
        """
        Convert DEVICE_TIME timestamp into TIME_CODE in nanoseconds.
        """
    def convert_from_synctime_to_device_time_ns(self, sync_time_ns: int, mode: projectaria_tools.core.sensor_data.TimeSyncMode) -> int:
        """
        Convert sync timestamp into synchronized timestamp in nanoseconds.
        """
    def convert_from_timecode_to_device_time_ns(self, timecode_time_ns: int) -> int:
        """
        Convert TIME_CODE timestamp into DEVICE_TIME in nanoseconds.
        """
    @typing.overload
    def deliver_queued_sensor_data(self) -> typing.Iterator[projectaria_tools.core.sensor_data.SensorData]:
        """
        Delivers data from all sensors in the entire vrs file sorted by TimeDomain.DEVICE_TIME.
        """
    @typing.overload
    def deliver_queued_sensor_data(self, arg0: DeliverQueuedOptions) -> typing.Iterator[projectaria_tools.core.sensor_data.SensorData]:
        """
        Delivers data from vrs file with options sorted by TimeDomain.DEVICE_TIME.
        """
    def get_all_streams(self) -> list[projectaria_tools.core.stream_id.StreamId]:
        """
        Get all available streams from the vrs file.
        """
    def get_audio_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.AudioConfig:
        ...
    def get_audio_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> tuple[projectaria_tools.core.sensor_data.AudioData, projectaria_tools.core.sensor_data.AudioDataRecord]:
        ...
    def get_audio_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> tuple[projectaria_tools.core.sensor_data.AudioData, projectaria_tools.core.sensor_data.AudioDataRecord]:
        ...
    def get_barometer_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.BarometerConfigRecord:
        ...
    def get_barometer_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.BarometerData:
        ...
    def get_barometer_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.BarometerData:
        ...
    def get_bluetooth_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.BluetoothBeaconConfigRecord:
        ...
    def get_bluetooth_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.BluetoothBeaconData:
        ...
    def get_bluetooth_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.BluetoothBeaconData:
        ...
    def get_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.SensorConfiguration:
        """
        Get configuration of a specific stream.
        """
    def get_default_deliver_queued_options(self) -> DeliverQueuedOptions:
        """
        Default options that delivers all sensor data in vrs from start to end in TimeDomain.DEVICE_TIME.
        """
    def get_device_calibration(self) -> projectaria_tools.core.calibration.DeviceCalibration | None:
        """
        Get calibration of the device.
        """
    def get_file_tags(self) -> dict[str, str]:
        """
        Get the tags map from the vrs file.
        """
    def get_first_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_domain: projectaria_tools.core.sensor_data.TimeDomain) -> int:
        """
        Get first timestamp in nanoseconds of a stream_id at a particular timeDomain.
        """
    def get_first_time_ns_all_streams(self, time_domain: projectaria_tools.core.sensor_data.TimeDomain) -> int:
        """
        Get first timestamp in nanoseconds of all stream_ids at a particular timeDomain.
        """
    def get_gps_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.GpsConfigRecord:
        ...
    def get_gps_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.GpsData:
        ...
    def get_gps_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.GpsData:
        ...
    def get_image_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.ImageConfigRecord:
        ...
    def get_image_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> tuple[projectaria_tools.core.sensor_data.ImageData, projectaria_tools.core.sensor_data.ImageDataRecord]:
        ...
    def get_image_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> tuple[projectaria_tools.core.sensor_data.ImageData, projectaria_tools.core.sensor_data.ImageDataRecord]:
        ...
    def get_imu_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.MotionConfigRecord:
        ...
    def get_imu_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.MotionData:
        ...
    def get_imu_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.MotionData:
        ...
    def get_index_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> int:
        """
        Get index of a the data from query timestamp in nanoseconds.
        """
    def get_label_from_stream_id(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> str | None:
        """
        Get label from stream_id as opposed to get_stream_id_from_label().
        """
    def get_last_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_domain: projectaria_tools.core.sensor_data.TimeDomain) -> int:
        """
        Get last timestamp in nanoseconds of a stream_id at a particular timeDomain.
        """
    def get_last_time_ns_all_streams(self, time_domain: projectaria_tools.core.sensor_data.TimeDomain) -> int:
        """
        Get last timestamp in nanoseconds of all stream_ids at a particular timeDomain.
        """
    def get_magnetometer_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.MotionConfigRecord:
        ...
    def get_magnetometer_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.MotionData:
        ...
    def get_magnetometer_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.MotionData:
        ...
    def get_metadata(self) -> VrsMetadata | None:
        """
        Get metadata if the loaded file is a VRS file.
        """
    def get_nominalRateHz(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> float:
        """
        Gets the nominal frame rate in Hz of a specific stream.
        """
    def get_nominal_rate_hz(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> float:
        """
        Gets the nominal frame rate in Hz of a specific stream.
        """
    def get_num_data(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> int:
        """
        Return number of collected sensor data of a stream.
        """
    def get_sensor_calibration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.calibration.SensorCalibration | None:
        """
        Get calibration of a sensor from the device.
        """
    def get_sensor_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.SensorData:
        """
        Return the N-th data of a stream, return SensorData of NOT_VALID if out of range. see SensorData for more details on how sensor data are represented.
        """
    def get_sensor_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.SensorData:
        """
        Get sensorData from a specific timestamp in nanosecond from sensor stream_id.
        """
    def get_sensor_data_type(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.SensorDataType:
        """
        Get sensor_data_type from stream_id.
        """
    def get_stream_id_from_label(self, label: str) -> projectaria_tools.core.stream_id.StreamId | None:
        """
        Get stream_id from label as opposed to get_label_from_stream_id().
        """
    def get_time_sync_mode(self) -> MetadataTimeSyncMode | None:
        """
        Get time-sync mode if the loaded file is a VRS file.
        """
    def get_timestamps_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_domain: projectaria_tools.core.sensor_data.TimeDomain) -> list[int]:
        """
        Get all timestamps in nanoseconds as a vector for stream_id of a particular timeDomain.
        """
    def get_wps_configuration(self, stream_id: projectaria_tools.core.stream_id.StreamId) -> projectaria_tools.core.sensor_data.WifiBeaconConfigRecord:
        ...
    def get_wps_data_by_index(self, stream_id: projectaria_tools.core.stream_id.StreamId, index: int) -> projectaria_tools.core.sensor_data.WifiBeaconData:
        ...
    def get_wps_data_by_time_ns(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_ns: int, time_domain: projectaria_tools.core.sensor_data.TimeDomain, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> projectaria_tools.core.sensor_data.WifiBeaconData:
        ...
    def supports_time_domain(self, stream_id: projectaria_tools.core.stream_id.StreamId, time_domain: projectaria_tools.core.sensor_data.TimeDomain) -> bool:
        """
        Check if a stream contains timestamp of a specific time domain specifically, Audio, Barometer, and GPS data does not have host timestamps. if the vrs does not contain a timesync stream with timecode mode, then timecode is not supported.
        """
class VrsMetadata:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def device_id(self) -> str:
        ...
    @property
    def device_serial(self) -> str:
        ...
    @property
    def duration_sec(self) -> int:
        ...
    @property
    def end_time_epoch_sec(self) -> int:
        ...
    @property
    def filename(self) -> str:
        ...
    @property
    def recording_profile(self) -> str:
        ...
    @property
    def shared_session_id(self) -> str:
        ...
    @property
    def start_time_epoch_sec(self) -> int:
        ...
    @property
    def time_sync_mode(self) -> MetadataTimeSyncMode:
        ...
def create_vrs_data_provider(vrs_filename: str) -> ...:
    """
    Factory class to create a VrsDataProvider class.
    """
