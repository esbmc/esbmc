# None on the then-branch: None if cond else T
f = lambda n: None if n < 0 else n * 2
x = f(-1)
assert x is None

y = f(3)
assert y is not None
