def f() -> str:
    return "x"


d = {"a": f}
assert d["a"]() == "y"
