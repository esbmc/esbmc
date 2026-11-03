if nondet_bool():
    a = 1
else:
    a = "ab"
try:
    h = a / 2
except TypeError:
    h = 0
try:
    d = a - 2
except TypeError:
    d = 0
try:
    c = a < 2
except TypeError:
    c = False
caught = False
try:
    s = a + "x"
except TypeError:
    caught = True
assert caught == isinstance(a, int)
