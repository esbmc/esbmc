x = 23
y = 7

result = divmod(x, y)
q = result[0]
r = result[1]

assert q == 3
assert r == 2
assert x == q * y + r  # Verify divmod property
