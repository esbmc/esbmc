def f() -> None:
    d = {}
    try:
        raise ValueError()
    except ValueError:
        d["a"] = 5
    d["b"] = []
    x = d["a"]
    assert x == 5

f()