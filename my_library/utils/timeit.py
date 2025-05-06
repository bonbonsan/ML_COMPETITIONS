import os
import time
from functools import wraps

from dotenv import load_dotenv

from my_library.utils.logger import Logger

logger = Logger(__name__).get_logger()

load_dotenv()
ENABLE_TIMEIT = os.getenv("ENABLE_TIMEIT", "True") == "True"


def timeit(func):
    """
    A decorator that logs the execution time of a function or method.

    This is useful for profiling performance-critical parts of a machine learning library.

    Example:
        from timeit import timeit

        @timeit
        def train_model():
            # Simulate training
            time.sleep(1.5)
            return "done"

        class Trainer:
            @timeit
            def run(self):
                time.sleep(0.5)

    Args:
        func (Callable): The target function or method.

    Returns:
        Callable: The wrapped function with execution time logging.
    """
    if not ENABLE_TIMEIT:
        return func
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[Start] {func.__qualname__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"[End] {func.__qualname__} - Elapsed: {elapsed_time:.4f} sec")
        return result
    return wrapper


if __name__ == "__main__":
    @timeit
    def slow_function():
        time.sleep(1.0)

    class SampleClass:
        @timeit
        def method(self):
            time.sleep(0.5)

    slow_function()
    SampleClass().method()
