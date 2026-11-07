def mk():
    a = [0, 1]
    a[0] = [4]
    return a


assert mk() == [[4], 1]
assert mk() != [[5], 1]
