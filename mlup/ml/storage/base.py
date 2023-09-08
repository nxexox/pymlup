import abc
from typing import Union, List, Callable

from mlup.constants import LoadedFile


class BaseStorage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load_bytes_single_file(self, path_to_file: str) -> Union[str, bytes, Callable]:
        pass

    @abc.abstractmethod
    def load(self) -> List[LoadedFile]:
        pass
