# Issue #8977: a function returning an int or a nondet str.
def elem():
    if nondet_bool():
        return 1
    return nondet_str()


xs = [1, 2]
xs.append(elem())
assert len(xs) == 3

x = elem()
ys = []
ys.append(x)
assert len(ys) == 1
