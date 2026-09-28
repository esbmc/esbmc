# bool() of a list, dict or tuple relabelled the container as a bool, which
# the solver rejected; a container is true when it is non-empty.
c = nondet_bool()
ys = [1]
if c:
    ys.pop()
assert bool(ys)
