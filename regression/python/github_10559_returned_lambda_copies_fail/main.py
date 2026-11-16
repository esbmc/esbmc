def func(k):
    return lambda z: z + k


y = func(1)
a = y(2)
y2 = func(1.5)
a2 = y2(0.5)
assert type(a) is float
