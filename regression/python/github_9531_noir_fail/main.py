u = nondet_bool()
d = "s" if u else nondet_int()
if not u:
    assert d >= 0
