# KeyError is not an ArithmeticError, so it escapes the handler.
d = {1: 2}
try:
    v = d[3]
except ArithmeticError:
    v = 0
assert v == 0
