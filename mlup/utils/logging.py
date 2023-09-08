import logging
from typing import Optional

from uvicorn.logging import DefaultFormatter

from mlup.utils.oop import MetaSingleton


class SwitchFormatter(DefaultFormatter, metaclass=MetaSingleton):
    def __init__(self, fmt: Optional[str] = None, *args, **kwargs):
        self._fmts = {'default': fmt}
        kwargs_without_fmt = {}
        for arg, val in kwargs.items():
            if arg.startswith('fmt_'):
                self._fmts[arg.split('_', 1)[1]] = val
            else:
                kwargs_without_fmt[arg] = val

        super().__init__(fmt=fmt, *args, **kwargs_without_fmt)

    def set_fmt(self, fmt_name: str = 'default'):
        fmt = self._fmts[fmt_name]
        self._style = logging.PercentStyle(fmt)
        self._style.validate()
        self._fmt = self._style._fmt


def configure_logging_formatter(formatter_name: str = 'default'):
    SwitchFormatter().set_fmt(formatter_name)
