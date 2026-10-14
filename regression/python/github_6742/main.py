import my_module


def abspath(n: int) -> int:
    return n + 1


assert abspath(1) == 2
assert my_module.abspath(1) == 101
