flag = nondet_bool()
x = "ab" if flag else 5
assert not isinstance(x, str) or x[0] == "a"
