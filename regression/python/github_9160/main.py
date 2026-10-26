import sys
import time

if sys.platform == "win32":
    clock = time.perf_counter
else:
    clock = time.time


def elapsed() -> float:
    start = clock()
    return clock() - start


def f(x: int) -> int:
    assert x != 4
    return x


f(3)
