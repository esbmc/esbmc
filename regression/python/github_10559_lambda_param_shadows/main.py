# The lambda's parameter x does not rebind the module's x.
def make(k):
    return lambda x: x + k


x = 3
f = make(2)
a = f(x)
assert type(a) is type(x)
assert a == 5
