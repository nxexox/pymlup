import time
from typing import Optional

from mlup.config import logger


class TimeProfiler:
    _start: float
    _end: float

    def __init__(self, msg: Optional[str] = None, log_level: str = 'info'):
        self.msg = msg
        self.logger = getattr(logger, log_level)

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.monotonic()
        msg = 'Time to func'
        if self.msg:
            msg = self.msg
        self.logger(
            f'{msg} {self._end - self._start:.3f}.'
        )
