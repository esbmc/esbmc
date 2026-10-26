def elem():
    if nondet_bool():
        return 1
    return nondet_str()


xs = [1, 2]
xs.append(elem())
assert len(xs) == 2
