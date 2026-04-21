def test() -> None:
    # Existing: inline dict.get with int defaults.
    x = {"a": 1}.get("missing", 3)
    assert x == 3

    y = {"a": 1}.get("a", 99)
    assert y == 1

    # Inline dict.get with scalar defaults of other builtin types: the
    # frontend must narrow the return type to the concrete builtin
    # instead of falling back to Any.
    assert {"a": 1}.get("missing", 3.14) > 3.0
    assert {"a": 1}.get("missing", "x") == "x"
    assert {"a": 1}.get("missing", True) == True

    # Untyped dict variable: same inference applies through the
    # converter's unannotated-assign path.
    d = {}
    assert d.get("k", 7) == 7

    z = list({}.items())
    assert len(z) == 0

test()
