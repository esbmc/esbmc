import functools


@functools.lru_cache
def f(a):
    return a


assert f(a=1, b=2) == 1
