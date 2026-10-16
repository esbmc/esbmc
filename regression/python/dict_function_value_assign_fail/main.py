def f() -> int:
    return 1


def g() -> int:
    return 2


def main() -> None:
    d = {"a": f}
    d["a"] = g
    assert d["a"]() == 2
    d.update({"a": f})
    assert d["a"]() == 1
    d["b"] = g
    assert d["b"]() == 1


main()
