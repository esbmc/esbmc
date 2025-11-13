x = -7
y = 3
quotient, remainder = divmod(x, y)

# Python's divmod follows floor division semantics:
# -7 // 3 = -3 (floor of -2.333...)
# -7 % 3 = 2 (because -7 = 3 * (-3) + 2)
assert quotient == -3
assert remainder == 2
