flag = nondet_bool()
v = "ab" if flag else 5
assert not isinstance(v, str) or v.startswith("a")
