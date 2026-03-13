# Lambda returning int or None based on condition
f = lambda n: n * 2 if n > 0 else None
x = f(1)
assert x is not None
