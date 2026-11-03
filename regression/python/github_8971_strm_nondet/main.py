flag = nondet_bool()
v = nondet_str() if flag else nondet_int()
if isinstance(v, str):
    starts = v.startswith("")
    assert starts
