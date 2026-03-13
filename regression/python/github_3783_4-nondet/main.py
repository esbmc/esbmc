# popitem() returns the nondet value that was inserted last
v: int = nondet_int()
d: dict[str, int] = {"a": 1, "b": v}
key, value = d.popitem()
assert key == "b"
assert value == v
