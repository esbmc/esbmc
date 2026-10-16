if nondet_bool():
    a = 1
else:
    a = "ab"
try:
    h = a - 2
except ValueError:
    h = 0
