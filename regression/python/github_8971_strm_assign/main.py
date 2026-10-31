flag = nondet_bool()
v = "ab" if flag else 5
if isinstance(v, str):
    starts = v.startswith("a")
    low = v.lower()
    digit = v.isdigit()
    assert starts
    assert low == "ab"
    assert not digit
