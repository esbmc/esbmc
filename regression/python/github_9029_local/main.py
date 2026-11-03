def call(h):
    if h == 0:
        r = "a"
    else:
        r = 1
    return r


assert call(0) == "a"
assert call(1) == 1
