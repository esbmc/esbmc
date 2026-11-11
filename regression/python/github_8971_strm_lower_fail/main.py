flag = nondet_bool()
v = "ab" if flag else 5
if isinstance(v, str):
    low = v.lower()
    assert low == "AB"
