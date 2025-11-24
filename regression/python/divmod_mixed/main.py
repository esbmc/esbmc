# test_divmod_mixed.py
x = 10
y = 3.0
quotient, remainder = divmod(x, y)

assert quotient == 3.0
assert remainder == 1.0
