import logging
import os
import re
from dataclasses import dataclass
from typing import Union, List

from mlup.ml.storage.base import BaseStorage
from mlup.constants import LoadedFile


logger = logging.getLogger('mlup')


@dataclass
class DiskStorage(BaseStorage):
    path_to_files: str
    files_mask: Union[str, re.Pattern] = re.compile(r'(\w.-_)*.pckl')
    need_load_file: bool = False

    def load_bytes_single_file(self, path_to_file: str) -> LoadedFile:
        logger.info(f'Load file {path_to_file}')
        if self.need_load_file and os.path.isfile(path_to_file):
            with open(path_to_file, 'rb') as f:
                return LoadedFile(f.read(), path=path_to_file)
        logger.debug(f'Return path {path_to_file} to binarizer')
        return LoadedFile(path=path_to_file)

    def load(self) -> List[LoadedFile]:
        logger.debug(f'Start analyze data path {self.path_to_files} by mask {self.files_mask}')
        if os.path.isfile(self.path_to_files):
            logger.debug(f'Found single file by path {self.path_to_files}')
            if re.search(self.files_mask, str(self.path_to_files)):
                return [self.load_bytes_single_file(self.path_to_files)]
            raise FileNotFoundError(f'File {self.path_to_files} does not match mask {self.files_mask}')

        file_pathes = [
            os.path.join(self.path_to_files, f_name)
            for f_name in os.listdir(self.path_to_files)
            if re.search(self.files_mask, f_name)
        ]
        logger.debug('Found files for loading: ' + ', '.join(file_pathes))
        return [self.load_bytes_single_file(f_path) for f_path in file_pathes]
