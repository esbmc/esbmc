def f() -> None:
    d = {}
    d["a"] = 1.5
    d["b"] = None
    x = d["a"]
    assert x == 1.5

f()
