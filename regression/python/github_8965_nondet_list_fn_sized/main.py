def flag():
    return "yes"


xs = nondet_list(2, flag)
ys = nondet_list(max_size=2, elem_type=flag)
assert len(xs) <= 2
for v in xs:
    assert v == "yes"
for v in ys:
    assert v == "yes"
