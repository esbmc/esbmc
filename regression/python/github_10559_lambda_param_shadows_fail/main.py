def make(k):
    return lambda x: x + k


x = 3
f = make(2)
a = f(x)
assert type(a) is float
