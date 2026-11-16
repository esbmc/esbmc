xs = nondet_list(3, elem_type=nondet_float())
for i, x in enumerate(xs):
    if not isinstance(x, float):
        assert False
