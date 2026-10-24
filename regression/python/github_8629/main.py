a = [nondet_int(), nondet_float(), nondet_str()]
b, c, d = a
assert isinstance(b, int)
assert isinstance(c, float)
assert isinstance(d, str)
