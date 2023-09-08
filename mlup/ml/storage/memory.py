from dataclasses import dataclass
from typing import List, Callable

from mlup.ml.storage.base import BaseStorage
from mlup.constants import LoadedFile


@dataclass
class MemoryStorage(BaseStorage):
    model: Callable

    def load_bytes_single_file(self, *args, **kwargs) -> Callable:
        return self.model

    def load(self) -> List[LoadedFile]:
        return [LoadedFile(self.model)]
