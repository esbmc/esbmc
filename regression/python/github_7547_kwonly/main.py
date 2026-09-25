def f(*, k: int) -> int:
    return k


def g() -> int:
    return f()
