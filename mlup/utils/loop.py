import asyncio
import nest_asyncio
import sys


# TODO: Fix event loop
nest_asyncio.apply()
# TODO: Does we support Python <= 3.7.0? Check tests
if sys.version_info >= (3, 7, 0):
    get_running_loop = asyncio.get_running_loop
else:
    def get_running_loop() -> asyncio.AbstractEventLoop:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            raise RuntimeError('no running event loop')
        return loop
