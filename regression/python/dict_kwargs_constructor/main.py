def main() -> None:
    # dict(a=1, b=2) (keyword form) and dict() (empty) previously aborted with
    # "Projecting from non-tuple based AST": the dict() constructor only handled
    # a single positional iterable, so the zero-positional-arg forms fell through
    # to a malformed lowering. Build the dict from the keyword arguments instead.
    d = dict(a=1, b=2)
    assert d["a"] == 1 and d["b"] == 2
    assert "a" in d and "z" not in d
    assert len(d) == 2

    # Empty dict(), then mutate.
    e = dict()
    assert len(e) == 0
    e["x"] = 5
    assert e["x"] == 5

    # The positional-iterable form is unchanged.
    f = dict([("k", 9)])
    assert f["k"] == 9


main()
