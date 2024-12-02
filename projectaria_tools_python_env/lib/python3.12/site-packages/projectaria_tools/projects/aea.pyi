from __future__ import annotations
import projectaria_tools.core.data_provider
import projectaria_tools.core.mps
import projectaria_tools.core.sensor_data
import typing
__all__ = ['AriaEverydayActivitiesDataPaths', 'AriaEverydayActivitiesDataPathsProvider', 'AriaEverydayActivitiesDataProvider', 'SentenceData', 'SpeechDataProvider', 'WordData']
class AriaEverydayActivitiesDataPaths:
    """
    A struct that includes the file paths of all AEA data files for one sequence
    """
    aria_vrs: str
    metadata: str
    mps: projectaria_tools.core.mps.MpsDataPaths
    speech: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class AriaEverydayActivitiesDataPathsProvider:
    """
    This class is to load all data file paths from AEA data structure given a sequence path. Each AEA collection sequence can only contain one Aria device and its associated data: 
    ├── sequencePath
    │   ├── metadata.json
    │   ├── recording.vrs
    │   ├── speech.csv
    │   ├── mps
    │   │   ├── {SEE MpsDataPathsProvider}
    This class allows you use dataset root to query all data associated with a single device recording.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str) -> None:
        ...
    def get_concurrent_recordings(self) -> list[str]:
        """
        Get the name(s) of the recording(s) that were collected at the same time and location as the current recording. This data can be found in the metadata json file
        """
    def get_data_paths(self) -> AriaEverydayActivitiesDataPaths:
        """
        Get the resulting data paths which can be fed to the AEA data provider to load the data
        """
    def get_dataset_name(self) -> str:
        """
        Get the name of the current dataset (AEA)
        """
    def get_dataset_version(self) -> str:
        """
        Get the version of the current dataset (AEA)
        """
    def get_location_number(self) -> int:
        """
        Get the location number. This number is found in the metadata json file, and is also embedded in the sequence name
        """
    def get_recording_number(self) -> int:
        """
        Get the recording number. This number is found in the metadata json file, and is also embedded in the sequence name
        """
    def get_script_number(self) -> int:
        """
        Get the script number. This number is found in the metadata json file, and is also embedded in the sequence name
        """
    def get_sequence_number(self) -> int:
        """
        Get the sequence number. This number is found in the metadata json file, and is also embedded in the sequence name
        """
class AriaEverydayActivitiesDataProvider:
    """
    This is the core data loader that provide all assets for an AEA sequence
    """
    mps: projectaria_tools.core.mps.MpsDataProvider
    speech: ...
    vrs: projectaria_tools.core.data_provider.VrsDataProvider
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: AriaEverydayActivitiesDataPaths) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    def has_aria_data(self) -> bool:
        ...
    def has_mps_data(self) -> bool:
        ...
    def has_speech_data(self) -> bool:
        ...
class SentenceData:
    """
    A simple struct to store data associated with one sentence from the speech to text data. Sentences are broken up by punctuation marks including: .  !  ?
    """
    end_timestamp_ns: int
    start_timestamp_ns: int
    words: dict[int, WordData]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
    def to_string(self) -> str:
        """
        Return the full sentence as a single string. Same as calling str() on the object
        """
class SpeechDataProvider:
    """
    Class for reading and querying speech data generated in the AEA sequence
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str) -> None:
        ...
    def empty(self) -> bool:
        ...
    def get_sentence_data_by_timestamp_ns(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> SentenceData | None:
        """
        Get sentence data given a query timestamp. A sentence is a series of words, where each sentence has a start and end timestamp, and each word has a start and end timestamp.Note: TimeQueryOptions is ignored if there is a sentence that contains the query device timestamp.
        """
    def get_word_data_by_timestamp_ns(self, device_timestamp_ns: int, time_query_options: projectaria_tools.core.sensor_data.TimeQueryOptions = ...) -> WordData | None:
        """
        Get word data given a query timestamp. A word has a start and end timestamp and confidence level. Note: TimeQueryOptions is ignored if there is a word that contains the query device timestamp.
        """
class WordData:
    """
    A simple struct to store data associated with one word from the speech to text data
    """
    confidence: float
    end_timestamp_ns: int
    start_timestamp_ns: int
    word: str
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
