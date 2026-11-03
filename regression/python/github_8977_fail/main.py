def elem():
    if nondet_bool():
        return 1
    return nondet_str()


x = elem()
assert x != 1
