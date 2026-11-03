def get(x):
    return x


v = "hello"
w = "world"
b = get(w)
assert not (b is v)
assert b is w


assert get(v) is v
