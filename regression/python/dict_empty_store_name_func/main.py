def f() -> None:
    x = 1
    d = {}
    d["a"] = x
    assert isinstance(d["a"], int)


f()
