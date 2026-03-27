def func() -> int:
    return 2


def call_func() -> int:
    return func()


def call_call_func() -> int:
    return call_func()


assert call_call_func() == 2
