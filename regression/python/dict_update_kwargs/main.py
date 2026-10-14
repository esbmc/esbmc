def main() -> None:
    # dict.update(k=v, ...) (keyword form) and dict.update() (empty) previously
    # raised "update() takes exactly one argument": update only accepted a single
    # positional source. Apply the keyword pairs (after any positional source).
    d = {"a": 1}
    d.update(b=2, c=3)
    assert d["b"] == 2 and d["c"] == 3

    # Keyword pair overwrites an existing key.
    d.update(a=9)
    assert d["a"] == 9

    # Empty update() is a no-op; the positional + keyword form together work.
    e = {"x": 1}
    e.update()
    e.update({"y": 2}, z=3)
    assert e["x"] == 1 and e["y"] == 2 and e["z"] == 3


main()
