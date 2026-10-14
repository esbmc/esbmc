def g():
    return a


a = 1
a += 1
assert g() == 2
