assert int is int  # True (direct comparison)
x = int
assert x is int  # True (variable vs direct)
y = int
assert x is y  # True (both variables hold same type)
z = str
assert x is not z  # True (different types with IsNot)
