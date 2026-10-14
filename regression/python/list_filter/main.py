# list(filter(pred, seq)) must materialise the kept elements in order,
# matching CPython. Exercises the three predicate forms the rewrite handles:
# a lambda, a named function, and filter(None, ...) (truthiness).


def pos(n: int) -> bool:
    return n > 0


# lambda predicate
xs = [1, -2, 3, -4]
kept = list(filter(lambda x: x > 0, xs))
assert len(kept) == 2
assert kept[0] == 1
assert kept[1] == 3

# named-function predicate
ys = [-1, 5, -3, 7]
pos_ys = list(filter(pos, ys))
assert len(pos_ys) == 2
assert pos_ys[0] == 5
assert pos_ys[1] == 7

# filter(None, ...) keeps the truthy elements
zs = [0, 1, 0, 2]
truthy = list(filter(None, zs))
assert len(truthy) == 2
assert truthy[0] == 1
assert truthy[1] == 2
