# #10559: the int copy's lambda returns an int, not a float.
def func(k):
    return lambda x: x + k


y = func(1)
a = y(2)
assert type(a) is float
y2 = func(1.5)
a2 = y2(2.5)
assert a2 == 4.0
