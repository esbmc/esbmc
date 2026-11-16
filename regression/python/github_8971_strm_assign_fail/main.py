flag = nondet_bool()
v = "ab" if flag else 5
if isinstance(v, str):
    starts = v.startswith("b")
    assert starts
