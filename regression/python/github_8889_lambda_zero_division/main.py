f = lambda x: 10 // x
g = lambda x: 10 % x
h = lambda x: 10 / x
k = lambda x: 0 if x == 0 else 10 // x

caught = 0
try:
    f(0)
except ZeroDivisionError:
    caught += 1
try:
    g(0)
except ZeroDivisionError:
    caught += 1
try:
    h(0)
except ZeroDivisionError:
    caught += 1

assert caught == 3
assert f(3) == 3
assert g(3) == 1
assert h(4) == 2.5
assert k(0) == 0
