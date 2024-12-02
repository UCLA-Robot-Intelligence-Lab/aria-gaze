from __future__ import annotations
import numpy
import typing
__all__ = ['BILINEAR', 'Image3U8', 'ImageF32', 'ImageU16', 'ImageU64', 'ImageU8', 'InterpolationMethod', 'ManagedImage3U8', 'ManagedImageF32', 'ManagedImageU16', 'ManagedImageU64', 'ManagedImageU8', 'NEAREST_NEIGHBOR', 'debayer']
class Image3U8:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ImageF32:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ImageU16:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ImageU64:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ImageU8:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class InterpolationMethod:
    """
    Image interpolation method.
    
    Members:
    
      NEAREST_NEIGHBOR
    
      BILINEAR
    """
    BILINEAR: typing.ClassVar[InterpolationMethod]  # value = <InterpolationMethod.BILINEAR: 1>
    NEAREST_NEIGHBOR: typing.ClassVar[InterpolationMethod]  # value = <InterpolationMethod.NEAREST_NEIGHBOR: 0>
    __members__: typing.ClassVar[typing.Dict[str, InterpolationMethod]]  # value = {'NEAREST_NEIGHBOR': <InterpolationMethod.NEAREST_NEIGHBOR: 0>, 'BILINEAR': <InterpolationMethod.BILINEAR: 1>}
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
    def value(arg0: InterpolationMethod) -> int:
        ...
class ManagedImage3U8:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ManagedImageF32:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ManagedImageU16:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ManagedImageU64:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
class ManagedImageU8:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def at(self, x: int, y: int, channel: int = ...) -> float | int | int | int | ...:
        """
        Returns the pixel at (x, y, channel)
        """
    def get_height(self) -> int:
        """
        Returns the number of rows
        """
    def get_width(self) -> int:
        """
        Returns the number of columns
        """
    def to_numpy_array(self) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
        """
        Converts to numpy array
        """
def debayer(arg0: numpy.ndarray[numpy.uint8]) -> numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.uint8] | numpy.ndarray[numpy.uint16] | numpy.ndarray[numpy.uint64] | numpy.ndarray[...]:
    """
    Debayer and also correct color by preset color calibration
    """
BILINEAR: InterpolationMethod  # value = <InterpolationMethod.BILINEAR: 1>
NEAREST_NEIGHBOR: InterpolationMethod  # value = <InterpolationMethod.NEAREST_NEIGHBOR: 0>
