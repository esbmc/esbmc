def foo(start_index: int):
    def wrap(func: int) -> int:
        return func

    return wrap


f = foo(0)
assert f(1) == 1
