def f() -> None:
    d = {}
    d["k"] = True
    x = d["k"]
    assert x == True

f()
