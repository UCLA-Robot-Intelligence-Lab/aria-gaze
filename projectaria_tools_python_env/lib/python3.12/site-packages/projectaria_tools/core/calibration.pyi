from __future__ import annotations
import projectaria_tools.core.image
import numpy
import pybind11_stubgen.typing_ext
import typing
__all__ = ['ARIA_ET_CALIBRATION', 'ARIA_MIC_CALIBRATION', 'BAROMETER_CALIBRATION', 'BarometerCalibration', 'CAMERA_CALIBRATION', 'CameraCalibration', 'CameraModelType', 'CameraProjection', 'DeviceCadExtrinsics', 'DeviceCalibration', 'FISHEYE624', 'IMU_CALIBRATION', 'ImuCalibration', 'KANNALA_BRANDT_K3', 'LINEAR', 'LinearRectificationModel3d', 'MAGNETOMETER_CALIBRATION', 'MICROPHONE_CALIBRATION', 'MagnetometerCalibration', 'MicrophoneCalibration', 'NOT_VALID', 'SPHERICAL', 'SensorCalibration', 'SensorCalibrationType', 'device_calibration_from_json', 'device_calibration_from_json_string', 'distort_by_calibration', 'distort_depth_by_calibration', 'distort_label_by_calibration', 'get_linear_camera_calibration', 'get_spherical_camera_calibration', 'rotate_camera_calib_cw90deg']
class BarometerCalibration:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str, arg1: float, arg2: float) -> None:
        ...
    def get_label(self) -> str:
        ...
    def raw_to_rectified(self, raw: float) -> float:
        ...
    def rectified_to_raw(self, rectified: float) -> float:
        ...
class CameraCalibration:
    """
    A class that provides APIs for camera calibration, including extrinsics, intrinsics, and projection.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: CameraModelType, arg2: numpy.ndarray[numpy.float64[m, 1]], arg3: SE3, arg4: int, arg5: int, arg6: float | None, arg7: float, arg8: str) -> None:
        """
        Constructor with a list of parameters for CameraCalibration.
          Args:
            label: The label of the camera, e.g. "camera-slam-left".
            projection_model_type The type of camera projection model, e.g. ModelType::Linear
            T_Device_Camera: The extrinsics of camera in Device frame.
            image_width: Width of camera image.
            image_height: Height of camera image.
            maybe_valid_radius: [optional] radius of a circular mask that represents the valid area on
                    the camera's sensor plane. Pixels out of this circular region are considered invalid. Setting
                    this to None means the entire sensor plane is valid.
            max_solid_angle: an angle theta representing the FOV cone of the camera. Rays out of
                    [-theta, +theta] will be rejected during projection
            serial_number: Serial number of the camera.
        """
    @typing.overload
    def __init__(self, arg0: str, arg1: CameraModelType, arg2: numpy.ndarray[numpy.float64[m, 1]], arg3: SE3, arg4: int, arg5: int, arg6: float | None, arg7: float, arg8: str, arg9: float) -> None:
        """
        Constructor with a list of parameters for CameraCalibration.
          Args:
            label: The label of the camera, e.g. "camera-slam-left".
            projection_model_type The type of camera projection model, e.g. ModelType::Linear
            T_Device_Camera: The extrinsics of camera in Device frame.
            image_width: Width of camera image.
            image_height: Height of camera image.
            maybe_valid_radius: [optional] radius of a circular mask that represents the valid area on
                    the camera's sensor plane. Pixels out of this circular region are considered invalid. Setting
                    this to None means the entire sensor plane is valid.
            max_solid_angle: an angle theta representing the FOV cone of the camera. Rays out of
                    [-theta, +theta] will be rejected during projection
            serial_number: Serial number of the camera.
            time_offset_sec_device_camera: time offset in second between the camera mid exposure time and the capture 
            timestamp.
        """
    @typing.overload
    def __init__(self, arg0: str, arg1: CameraModelType, arg2: numpy.ndarray[numpy.float64[m, 1]], arg3: SE3, arg4: int, arg5: int, arg6: float | None, arg7: float, arg8: str, arg9: float, arg10: float | None) -> None:
        """
        Constructor with a list of parameters for CameraCalibration.
          Args:
            label: The label of the camera, e.g. "camera-slam-left".
            projection_model_type The type of camera projection model, e.g. ModelType::Linear
            T_Device_Camera: The extrinsics of camera in Device frame.
            image_width: Width of camera image.
            image_height: Height of camera image.
            maybe_valid_radius: [optional] radius of a circular mask that represents the valid area on
                    the camera's sensor plane. Pixels out of this circular region are considered invalid. Setting
                    this to None means the entire sensor plane is valid.
            max_solid_angle: an angle theta representing the FOV cone of the camera. Rays out of
                    [-theta, +theta] will be rejected during projection
            serial_number: Serial number of the camera.
            time_offset_sec_device_camera: time offset in second between the camera mid exposure time and the capture 
            timestamp.
            maybe_readout_time_sec: readout time in second to read from the first pixel to the last pixel.
        """
    def __repr__(self) -> str:
        ...
    def get_focal_lengths(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        ...
    def get_image_size(self) -> numpy.ndarray[numpy.int32[2, 1]]:
        ...
    def get_label(self) -> str:
        ...
    def get_max_solid_angle(self) -> float:
        ...
    def get_principal_point(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        ...
    def get_readout_time_sec(self) -> float | None:
        ...
    def get_serial_number(self) -> str:
        ...
    def get_time_offset_sec_device_camera(self) -> float:
        ...
    def get_transform_device_camera(self) -> SE3:
        ...
    def get_valid_radius(self) -> float | None:
        ...
    def is_visible(self, camera_pixel: numpy.ndarray[numpy.float64[2, 1]]) -> bool:
        """
        Function to check whether a pixel is within the valid area of the sensor plane.
        """
    def model_name(self) -> CameraModelType:
        ...
    def project(self, point_in_camera: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[2, 1]] | None:
        """
        Function to project a 3d point (in camera frame) to a 2d camera pixel location, with a number of validity checks to ensure the point is visible.
        """
    def project_no_checks(self, point_in_camera: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        Function to project a 3d point (in camera frame) to a 2d camera pixel location. In this function, no check is performed.
        """
    def projection_params(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def rescale(self, new_resolution: numpy.ndarray[numpy.int32[2, 1]], scale: float, origin_offset: numpy.ndarray[numpy.float64[2, 1]] = ...) -> CameraCalibration:
        """
        Obtain a new camera calibration after translation and scaling transform from the original camera calibration. <br> transform is done in the order of (1) shift -> (2) scaling: new_resolution = (old_resolution - origin_offset*2) * scale
        """
    def unproject(self, camera_pixel: numpy.ndarray[numpy.float64[2, 1]]) -> numpy.ndarray[numpy.float64[3, 1]] | None:
        """
        Function to unproject a 2d pixel location to a 3d ray, in camera frame, with a number of validity checks to ensure the unprojection is valid.
        """
    def unproject_no_checks(self, camera_pixel: numpy.ndarray[numpy.float64[2, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Function to unproject a 2d pixel location to a 3d ray in camera frame. In this function, no check is performed.
        """
class CameraModelType:
    """
    Enum that represents the type of camera projection model. See Linear.h, Spherical.h, KannalaBrandtK3.h and FisheyeRadTanThinPrism.h for details.
    
    Members:
    
      KANNALA_BRANDT_K3 : Spherical + polynomial radial distortion up to 9-th order.
    
      FISHEYE624 : Spherical + polynomial radial distortion up to 11-th order + tangential distortion.
    
      SPHERICAL : Spherical projection, linear in angular space.
    
      LINEAR : Linear pinhole projection, unit plane points and camera pixels are linearly related.
    """
    FISHEYE624: typing.ClassVar[CameraModelType]  # value = <CameraModelType.FISHEYE624: 3>
    KANNALA_BRANDT_K3: typing.ClassVar[CameraModelType]  # value = <CameraModelType.KANNALA_BRANDT_K3: 2>
    LINEAR: typing.ClassVar[CameraModelType]  # value = <CameraModelType.LINEAR: 0>
    SPHERICAL: typing.ClassVar[CameraModelType]  # value = <CameraModelType.SPHERICAL: 1>
    __members__: typing.ClassVar[typing.Dict[str, CameraModelType]]  # value = {'KANNALA_BRANDT_K3': <CameraModelType.KANNALA_BRANDT_K3: 2>, 'FISHEYE624': <CameraModelType.FISHEYE624: 3>, 'SPHERICAL': <CameraModelType.SPHERICAL: 1>, 'LINEAR': <CameraModelType.LINEAR: 0>}
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
    def value(arg0: CameraModelType) -> int:
        ...
class CameraProjection:
    """
    A struct to represent a camera projection instance, which is basically camera intrinsics. This struct stores the intrinsics parameters internally.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor, creates an empty CameraProjection instance.
        """
    @typing.overload
    def __init__(self, arg0: ..., arg1: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        """
        Constructor with a list of parameters for CameraProjection.
                  Args:
                    type: The type of projection model, e.g. ModelType::Linear.
                    projection_params: The projection parameters.
        """
    def get_focal_lengths(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        returns focal lengths as {fx, fy}.
        """
    def get_principal_point(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        returns principal point location as {cx, cy}.
        """
    def model_name(self) -> ...:
        ...
    def project(self, point_in_camera: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        projects a 3d world point in the camera space to a 2d pixel in the image space. No checks performed in this process.
        """
    def projection_params(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def unproject(self, camera_pixel: numpy.ndarray[numpy.float64[2, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
         No checks performed in this process.
        """
class DeviceCadExtrinsics:
    """
    This class retrieves fixed CAD extrinsics values for Aria Device
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: str) -> None:
        """
        Construct for Cad extrinsics based on device sub type and origin label, where the label of the origin (`Device` coordinate frame) sensor,e.g. camera-slam-left
        """
class DeviceCalibration:
    """
    A class to store and access calibration information of a device, including: camera, imu, magnetometer, barometer, and microphones.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: dict[str, CameraCalibration], arg1: dict[str, ImuCalibration], arg2: dict[str, MagnetometerCalibration], arg3: dict[str, BarometerCalibration], arg4: dict[str, MicrophoneCalibration], arg5: DeviceCadExtrinsics, arg6: str, arg7: str) -> None:
        """
        Constructor that composes a collection of sensor calibrations into a DeviceCalibration"
           " @param camera_calibs: map of <label, CameraCalibration>"
           " @param imu_calibs: map of <label, ImuCalibration>"
           * @param magnetometer_calibs: map of <label, MagnetometerCalibration>
           * @param barometer_calibs: map of <label, BarometerCalibration>
           * @param microphone_calibs: map of <label, MicrophoneCalibration>
           * @param device_cad_extrinsics: a struct representing the CAD extrinsics info of the device sensors.
           * @param device_subtype: the subtype of the device. For Aria, this would be "DVT-S' or "DVT-L".
           * @param origin_label: the label identifying the origin of the calibration extrinsics, which needs
           to be a sensor within this device. This is basically the "Device" frame in `T_Device_Sensor`.
        """
    def get_all_labels(self) -> list[str]:
        """
        returns all labels for all the sensors.
        """
    def get_aria_et_camera_calib(self) -> typing.Annotated[list[CameraCalibration], pybind11_stubgen.typing_ext.FixedSize(2)] | None:
        """
        returns an array-of-2 of CameraCalibration representing left and right ET camera calibrations for an Aria device. Will return None if device is not Aria, or it does not contain the valid ET camera.
        """
    def get_aria_microphone_calib(self) -> typing.Annotated[list[MicrophoneCalibration], pybind11_stubgen.typing_ext.FixedSize(7)] | None:
        """
        returns an array-of-7 of mic calibration for an Aria device. Will return None if device is not Aria, or it does not contain the valid mic calibrations.
        """
    def get_barometer_calib(self, label: str) -> BarometerCalibration | None:
        """
        returns a barometer calibration by its label. Will return None if label does not exist in device calibration.
        """
    def get_barometer_labels(self) -> list[str]:
        """
        returns all labels for barometers.
        """
    def get_camera_calib(self, label: str) -> CameraCalibration | None:
        """
        returns a camera calibration by its label. Will return None if label does not exist in device calibration.
        """
    def get_camera_labels(self) -> list[str]:
        """
        returns all labels for cameras.
        """
    def get_device_subtype(self) -> str:
        """
        Get the subtype of device. For Aria, this is 'DVT-S' or 'DVT-L' to indicate the size of the Aria unit.
        """
    def get_imu_calib(self, label: str) -> ImuCalibration | None:
        """
        returns a imu calibration by its label. Will return None if label does not exist in device calibration.
        """
    def get_imu_labels(self) -> list[str]:
        """
        returns all labels for imus.
        """
    def get_magnetometer_calib(self, label: str) -> MagnetometerCalibration | None:
        """
        returns a magnetometer calibration by its label. Will return None if label does not exist in device calibration.
        """
    def get_magnetometer_labels(self) -> list[str]:
        """
        returns all labels for magnetometers.
        """
    def get_microphone_calib(self, label: str) -> MicrophoneCalibration | None:
        """
        returns a microphone calibration by its label. Will return None if label does not exist in device calibration.
        """
    def get_microphone_labels(self) -> list[str]:
        """
        returns all labels for microphones.
        """
    def get_origin_label(self) -> str:
        """
        obtain the definition of Origin (or Device in T_Device_Sensor).
        """
    def get_sensor_calib(self, label: str) -> SensorCalibration | None:
        """
        returns a sensor calibration by its label. Will return None if label does not exist in device calibration.
        """
    def get_transform_cpf_sensor(self, label: str, get_cad_value: bool = ...) -> SE3 | None:
        """
        returns calibrated sensor extrinsics in CPF frame given a label. You can return the CAD extrinsics value by specifying `get_cad_value = True`.
        """
    def get_transform_device_cpf(self) -> SE3:
        """
        returns relative pose between device frame (anchored to a particular sensor defined by `origin_label`) and CPF (central pupil frame), where CPF is a virtual coordinate frame defined in CAD model.
        """
    def get_transform_device_sensor(self, label: str, get_cad_value: bool = ...) -> SE3 | None:
        """
        returns calibrated `T_Device_Sensor` given a label. You can return the CAD extrinsics value by specifying `get_cad_value = True`.
        """
class ImuCalibration:
    """
    A class representing an IMU calibration model, including both accelerometer and gyroscope. We assume the accelerometer and gyroscope for each IMU are co-located and thus they share the same extrinsic.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str, arg1: numpy.ndarray[numpy.float64[3, 3]], arg2: numpy.ndarray[numpy.float64[3, 1]], arg3: numpy.ndarray[numpy.float64[3, 3]], arg4: numpy.ndarray[numpy.float64[3, 1]], arg5: SE3) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_accel_model(self) -> LinearRectificationModel3d:
        """
        Get accelerometer intrinsics model that contains rectification matrix and bias vector.
        """
    def get_gyro_model(self) -> LinearRectificationModel3d:
        """
        Get gyroscope intrinsics model that contains rectification matrix and bias vector.
        """
    def get_label(self) -> str:
        ...
    def get_transform_device_imu(self) -> SE3:
        ...
    def raw_to_rectified_accel(self, raw: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        convert from imu sensor readout to actual acceleration: rectified = rectificationMatrix.inv() * (raw - bias).
        """
    def raw_to_rectified_gyro(self, raw: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        convert from imu sensor readout to actual angular velocity: rectified = rectificationMatrix.inv() * (raw - bias).
        """
    def rectified_to_raw_accel(self, rectified: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        simulate imu accel sensor readout from actual acceleration: raw = rectificationMatrix * rectified + bias.
        """
    def rectified_to_raw_gyro(self, rectified: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        simulate imu gyro sensor readout from actual angular velocity:  raw = rectificationMatrix * rectified + bias.
        """
class LinearRectificationModel3d:
    """
    A class that contains imu and mag intrinsics rectification model.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: numpy.ndarray[numpy.float64[3, 3]], arg1: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    def get_bias(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the bias vector.
        """
    def get_rectification(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        """
        Get the rectification matrix. 
        """
class MagnetometerCalibration:
    """
    A class representing a magnetometer calibration model, including only the intrinsics of the magnetometer.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str, arg1: numpy.ndarray[numpy.float64[3, 3]], arg2: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    def get_label(self) -> str:
        ...
    def raw_to_rectified(self, raw: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        convert from mag sensor readout to actual magnetic field, rectified = rectificationMatrix.inv() * (raw - bias).
        """
    def rectified_to_raw(self, rectified: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        simulate mag sensor readout from actual magnetic field raw = rectificationMatrix * rectified + bias.
        """
class MicrophoneCalibration:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: float) -> None:
        ...
    def get_label(self) -> str:
        ...
    def raw_to_rectified(self, raw: float) -> float:
        ...
    def rectified_to_raw(self, rectified: float) -> float:
        ...
class SensorCalibration:
    """
    An adaptor class to access an arbitrary sensor's calibration, which is a python `enum` of {CameraCalibration, ImuCalibration, MagnetometerCalibration, BarometerCalibration, MicrophoneCalibration, AriaEtCalibration, AriaMicCalibration}
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: None | CameraCalibration | ImuCalibration | MagnetometerCalibration | BarometerCalibration | MicrophoneCalibration | typing.Annotated[list[CameraCalibration], pybind11_stubgen.typing_ext.FixedSize(2)] | typing.Annotated[list[MicrophoneCalibration], pybind11_stubgen.typing_ext.FixedSize(7)]) -> None:
        ...
    def aria_et_calibration(self) -> typing.Annotated[list[CameraCalibration], pybind11_stubgen.typing_ext.FixedSize(2)]:
        """
        Try to get the SensorCalibration as a AriaEtCalibration. Will throw if sensor type does not match.
        """
    def aria_mic_calibration(self) -> typing.Annotated[list[MicrophoneCalibration], pybind11_stubgen.typing_ext.FixedSize(7)]:
        """
        Try to get the SensorCalibration as a AriaMicCalibration. Will throw if sensor type does not match.
        """
    def barometer_calibration(self) -> BarometerCalibration:
        """
        Try to get the SensorCalibration as a BarometerCalibration. Will throw if sensor type does not match.
        """
    def camera_calibration(self) -> CameraCalibration:
        """
        Try to get the SensorCalibration as a CameraCalibration. Will throw if sensor type does not match.
        """
    def imu_calibration(self) -> ImuCalibration:
        """
        Try to get the SensorCalibration as a ImuCalibration. Will throw if sensor type does not match.
        """
    def magnetometer_calibration(self) -> MagnetometerCalibration:
        """
        Try to get the SensorCalibration as a MagnetometerCalibration. Will throw if sensor type does not match.
        """
    def microphone_calibration(self) -> MicrophoneCalibration:
        """
        Try to get the SensorCalibration as a MicrophoneCalibration. Will throw if sensor type does not match.
        """
    def sensor_calibration_type(self) -> SensorCalibrationType:
        """
        get the type of this sensor calibration as an enum.
        """
class SensorCalibrationType:
    """
    Members:
    
      NOT_VALID
    
      CAMERA_CALIBRATION
    
      IMU_CALIBRATION
    
      MAGNETOMETER_CALIBRATION
    
      BAROMETER_CALIBRATION
    
      MICROPHONE_CALIBRATION
    
      ARIA_ET_CALIBRATION
    
      ARIA_MIC_CALIBRATION
    """
    ARIA_ET_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.ARIA_ET_CALIBRATION: 6>
    ARIA_MIC_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.ARIA_MIC_CALIBRATION: 7>
    BAROMETER_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.BAROMETER_CALIBRATION: 4>
    CAMERA_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.CAMERA_CALIBRATION: 1>
    IMU_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.IMU_CALIBRATION: 2>
    MAGNETOMETER_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.MAGNETOMETER_CALIBRATION: 3>
    MICROPHONE_CALIBRATION: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.MICROPHONE_CALIBRATION: 5>
    NOT_VALID: typing.ClassVar[SensorCalibrationType]  # value = <SensorCalibrationType.NOT_VALID: 0>
    __members__: typing.ClassVar[typing.Dict[str, SensorCalibrationType]]  # value = {'NOT_VALID': <SensorCalibrationType.NOT_VALID: 0>, 'CAMERA_CALIBRATION': <SensorCalibrationType.CAMERA_CALIBRATION: 1>, 'IMU_CALIBRATION': <SensorCalibrationType.IMU_CALIBRATION: 2>, 'MAGNETOMETER_CALIBRATION': <SensorCalibrationType.MAGNETOMETER_CALIBRATION: 3>, 'BAROMETER_CALIBRATION': <SensorCalibrationType.BAROMETER_CALIBRATION: 4>, 'MICROPHONE_CALIBRATION': <SensorCalibrationType.MICROPHONE_CALIBRATION: 5>, 'ARIA_ET_CALIBRATION': <SensorCalibrationType.ARIA_ET_CALIBRATION: 6>, 'ARIA_MIC_CALIBRATION': <SensorCalibrationType.ARIA_MIC_CALIBRATION: 7>}
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
    def value(arg0: SensorCalibrationType) -> int:
        ...
def device_calibration_from_json(arg0: str) -> DeviceCalibration | None:
    """
    Load calibration from json.
    """
def device_calibration_from_json_string(arg0: str) -> DeviceCalibration | None:
    """
    Load calibration from json string.
    """
@typing.overload
def distort_by_calibration(arraySrc: numpy.ndarray[numpy.uint8], dstCalib: CameraCalibration, srcCalib: CameraCalibration, method: projectaria_tools.core.image.InterpolationMethod = ...) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image to swap its underlying image distortion model.
    """
@typing.overload
def distort_by_calibration(arraySrc: numpy.ndarray[numpy.float32], dstCalib: CameraCalibration, srcCalib: CameraCalibration, method: projectaria_tools.core.image.InterpolationMethod = ...) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image to swap its underlying image distortion model.
    """
@typing.overload
def distort_by_calibration(arraySrc: numpy.ndarray[numpy.uint16], dstCalib: CameraCalibration, srcCalib: CameraCalibration, method: projectaria_tools.core.image.InterpolationMethod = ...) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image to swap its underlying image distortion model.
    """
@typing.overload
def distort_by_calibration(arraySrc: numpy.ndarray[numpy.uint64], dstCalib: CameraCalibration, srcCalib: CameraCalibration, method: projectaria_tools.core.image.InterpolationMethod = ...) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image to swap its underlying image distortion model.
    """
@typing.overload
def distort_depth_by_calibration(arg0: numpy.ndarray[numpy.uint8], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input depth image using InterpolationMethod::Bilinear to swap its underlying image distortion model.
    """
@typing.overload
def distort_depth_by_calibration(arg0: numpy.ndarray[numpy.float32], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input depth image using InterpolationMethod::Bilinear to swap its underlying image distortion model.
    """
@typing.overload
def distort_depth_by_calibration(arg0: numpy.ndarray[numpy.uint16], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input depth image using InterpolationMethod::Bilinear to swap its underlying image distortion model.
    """
@typing.overload
def distort_depth_by_calibration(arg0: numpy.ndarray[numpy.uint64], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input depth image using InterpolationMethod::Bilinear to swap its underlying image distortion model.
    """
@typing.overload
def distort_label_by_calibration(arg0: numpy.ndarray[numpy.uint8], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image label using InterpolationMethod::NearestNeighbor to swap its underlying image distortion model.
    """
@typing.overload
def distort_label_by_calibration(arg0: numpy.ndarray[numpy.float32], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image label using InterpolationMethod::NearestNeighbor to swap its underlying image distortion model.
    """
@typing.overload
def distort_label_by_calibration(arg0: numpy.ndarray[numpy.uint16], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image label using InterpolationMethod::NearestNeighbor to swap its underlying image distortion model.
    """
@typing.overload
def distort_label_by_calibration(arg0: numpy.ndarray[numpy.uint64], arg1: CameraCalibration, arg2: CameraCalibration) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Distorts an input image label using InterpolationMethod::NearestNeighbor to swap its underlying image distortion model.
    """
def get_linear_camera_calibration(image_width: int, image_height: int, focal_length: float, label: str = ..., T_Device_Camera: SE3 = ..., time_offset_sec_device_camera: float = ...) -> CameraCalibration:
    """
    Function to create a simple Linear camera calibration object from some parameters.
    """
def get_spherical_camera_calibration(image_width: int, image_height: int, focal_length: float, label: str = ..., T_Device_Camera: SE3 = ..., time_offset_sec_device_camera: float = ...) -> CameraCalibration:
    """
    Function to create a simple Spherical camera calibration object from some parameters.
    """
def rotate_camera_calib_cw90deg(camera_calibration: CameraCalibration) -> CameraCalibration:
    """
    Rotate CameraCalibration (Linear model only) clock-wise for 90 degrees (Upright view)
    """
ARIA_ET_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.ARIA_ET_CALIBRATION: 6>
ARIA_MIC_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.ARIA_MIC_CALIBRATION: 7>
BAROMETER_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.BAROMETER_CALIBRATION: 4>
CAMERA_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.CAMERA_CALIBRATION: 1>
FISHEYE624: CameraModelType  # value = <CameraModelType.FISHEYE624: 3>
IMU_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.IMU_CALIBRATION: 2>
KANNALA_BRANDT_K3: CameraModelType  # value = <CameraModelType.KANNALA_BRANDT_K3: 2>
LINEAR: CameraModelType  # value = <CameraModelType.LINEAR: 0>
MAGNETOMETER_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.MAGNETOMETER_CALIBRATION: 3>
MICROPHONE_CALIBRATION: SensorCalibrationType  # value = <SensorCalibrationType.MICROPHONE_CALIBRATION: 5>
NOT_VALID: SensorCalibrationType  # value = <SensorCalibrationType.NOT_VALID: 0>
SPHERICAL: CameraModelType  # value = <CameraModelType.SPHERICAL: 1>
