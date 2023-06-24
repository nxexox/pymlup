import abc
from dataclasses import dataclass


class BaseMLupStorage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load_bytes_single_file(self):
        pass

    def save(self):
        pass


@dataclass
class LocalDiskMLupStorage(BaseMLupStorage):
    path_to_files: str

    def load_bytes_single_file(self):
        with open(self.path_to_files, 'rb') as f:
            return f.read()
