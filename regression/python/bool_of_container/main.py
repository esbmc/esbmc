# bool() of a list or dict relabelled the container as a bool, which the
# solver rejected; a container is true when it is non-empty.
def nonempty(xs: list) -> bool:
    return bool(xs)


xs = [1, 2]
assert bool(xs)
e: list = []
assert not bool(e)
d = {1: 2}
assert bool(d)
assert bool((1, )) and not ()
assert not nonempty([])
c = nondet_bool()
ys = [1]
if c:
    ys.pop()
assert bool(ys) == (not c)
