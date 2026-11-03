t: tuple[str, str] = ("x", nondet_str())
a, b = t
assert b != "ok"
