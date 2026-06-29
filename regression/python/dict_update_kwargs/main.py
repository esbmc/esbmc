def main() -> None:
    # dict.update() previously accepted only a single positional dict and
    # rejected keyword arguments ("update() takes exactly one argument").
    # update(a=1, b=2) now inserts each keyword as a string-keyed entry.
    d = {"a": 1}
    d.update(a=10, b=2)
    assert d["a"] == 10 and d["b"] == 2

    # No-argument update() is a no-op.
    e = {"x": 1}
    e.update()
    assert e["x"] == 1

    # A positional source plus keywords; keywords win on a key conflict.
    f = {}
    f.update({"a": 1}, b=2)
    assert f["a"] == 1 and f["b"] == 2
    g = {}
    g.update({"k": 1}, k=9)
    assert g["k"] == 9

    # Positional-only forms are unchanged.
    h = {"a": 1}
    h.update({"b": 2})
    assert h["a"] == 1 and h["b"] == 2


main()
