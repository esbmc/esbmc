def half(a: int) -> float:
    return a / 2


def foo(a: int) -> int:
    return half(a)


assert foo(3) == 1
