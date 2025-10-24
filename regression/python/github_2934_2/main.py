def foo(x: int | None = None) -> None:
    assert (x is None) == (x == None)  # is vs == should align for None

foo(None)
