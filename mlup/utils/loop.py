import asyncio
import sys
import threading
from concurrent.futures import Executor


class RunThreadCoro(threading.Thread):
    def __init__(self, async_func, args, kwargs):
        self.async_func = async_func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.async_func(*self.args, **self.kwargs))


def run_async(func, *args, **kwargs):
    # If loop is running, asyncio.run raise:
    # RuntimeError: asyncio.run() cannot be called from a running event loop
    # https://stackoverflow.com/questions/55409641/asyncio-run-cannot-be-called-from-a-running-event-loop-when-using-jupyter-no
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThreadCoro(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func(*args, **kwargs))


def create_async_task(coro, *, name=None):
    task_kwargs = {}
    if sys.version_info.minor >= 8:
        task_kwargs['name'] = name
    return asyncio.create_task(coro, **task_kwargs)


def shutdown_pool_executor(pool_executor: Executor, wait: bool = False, cancel_futures: bool = False):
    kwargs = {}
    if sys.version_info.minor >= 9:
        kwargs['cancel_futures'] = cancel_futures
    pool_executor.shutdown(wait, **kwargs)
