def mk():
    a = [0, 0]
    a[0:1] = [[7]]
    return a


assert mk() == [[7], 0]
assert mk() != [[8], 0]
