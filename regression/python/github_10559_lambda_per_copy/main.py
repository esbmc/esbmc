# #10559: each per-type copy of func returns its own lambda, typed by the
# calls through that copy's result alone.
def func(k):
    return lambda x: x + k


y = func(1)
a = y(2)
assert type(a) is int
assert a == 3
y2 = func(1.5)
a2 = y2(2.5)
assert type(a2) is float
assert a2 == 4.0
