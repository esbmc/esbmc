from functools import reduce


def add(x, y):
    return x + y


xs = [nondet_int()] + nondet_list(nondet_int, max_size=2)
p = reduce(add, xs)
assert isinstance(p, int)
