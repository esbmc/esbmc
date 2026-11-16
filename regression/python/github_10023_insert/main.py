def mk():
    a = [1]
    a.insert(0, [2])
    return a


assert mk() == [[2], 1]
assert mk() != [[3], 1]
