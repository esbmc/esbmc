# Contrast case for the dynamic_type_*_knownbug tests: list elements already
# use a tagged runtime representation (PyObject's value/type_id), so type correctly survives a
# branch join here.

cond = nondet_bool()
if cond:
    lst = [1]
else:
    lst = ["a"]
if not cond:
    assert lst[0] == "a"
    assert lst[0] != 1
