def foo(*, a: int, b: str | None = None):
    if b is not None and not isinstance(b, str):
        raise AssertionError("Python runtime sanity check")


foo(a=0)
