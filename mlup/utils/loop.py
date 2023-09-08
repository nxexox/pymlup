import asyncio
import threading


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
