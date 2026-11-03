# f is called with a float and with an int, so the lambda's parameter keeps
# its default float type rather than the type of either call.
def make():
    return lambda x: x * 2


f = make()
b = f(1.5)
a = f(3)
assert b == 3.0
assert a == 6
