xs = [nondet_int(), nondet_int()]
ys = [b for b in xs]
assert ys[0] == xs[0]
assert ys[1] == xs[1]
