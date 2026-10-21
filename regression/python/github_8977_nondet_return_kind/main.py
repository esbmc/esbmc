def elem():
    if nondet_bool():
        return 1
    return nondet_str()


def other():
    if nondet_bool():
        return nondet_int()
    return nondet_str()


xs = [1, 2]
xs.append(elem())
assert len(xs) == 3
x = other()
ys = []
ys.append(x)
assert len(ys) == 1
