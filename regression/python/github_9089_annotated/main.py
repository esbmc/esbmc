t: tuple[str, str] = ("x", nondet_str())
a, b = t
assert a == "x"
if b == "ok":
    assert len(b) == 2
