flag = nondet_bool()
x = "ab" if flag else 5
if isinstance(x, str):
    assert x[0:1] == "a"
