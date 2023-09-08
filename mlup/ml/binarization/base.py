import abc
from dataclasses import dataclass

from mlup.constants import LoadedFile


@dataclass(kw_only=True)
class BaseBinarizer(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: LoadedFile):
        pass

    @classmethod
    def is_this_type(cls, loaded_file: LoadedFile) -> float:
        return 0.0
