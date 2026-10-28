flag = nondet_bool()
v = "ab" if flag else 5
if isinstance(v, str):
    ends = v.endswith("b")
    assert ends
