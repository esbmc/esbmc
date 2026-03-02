def foo(x: None) -> None:
    assert x is None
    assert not (x is not None)


foo(None)
