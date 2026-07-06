# A lambda bound to a variable applies its default arguments when the argument
# is omitted (previously the parameter was modelled as nondet). Covers int,
# float, bool and negative defaults, multiple defaults, and partial
# application; an explicitly supplied argument overrides the default.
f = lambda x, y=2: x * y
assert f(3) == 6
assert f(3, 5) == 15

g = lambda x=10: x + 1
assert g() == 11

h = lambda a, b=2, c=3: a + b + c
assert h(1) == 6
assert h(1, 10) == 14

p = lambda x, y=1.5: x + y
assert p(2) == 3.5

q = lambda x, y=-5: x + y
assert q(10) == 5

b = lambda x, flag=True: flag
assert b(1) == True

# A lambda without defaults is unchanged.
n = lambda x, y: x * y
assert n(3, 2) == 6

# A def-bound function variable also fills scalar defaults through the same
# indirect-call path (a string default is left nondet, not asserted here).
def scaled(v, factor=3):
    return v * factor
op = scaled
assert op(4) == 12
assert op(4, 2) == 8
