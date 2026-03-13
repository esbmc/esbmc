# Lambda returns None when condition is false
f = lambda n: n * 2 if n > 0 else None
x = f(-1)
assert x is None
