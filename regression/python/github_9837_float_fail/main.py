def f() -> None:
    d = {}
    d["k"] = 1.5
    x = d["k"]
    assert x == 2.5

f()
