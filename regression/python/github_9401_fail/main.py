# #9401: float() strips only leading and trailing whitespace.
try:
    x = float(" 2.5 ")
except ValueError:
    x = 0.0
assert x == 0.0
