# func is copied per argument type; each copy's lambda takes the type of the
# calls through its own result.
def func(k):
    return lambda z: z + k


y = func(1)
a = y(2)
assert type(a) is int
y2 = func(1.5)
a2 = y2(0.5)
assert type(a2) is float
assert a == 3 and a2 == 2.0
