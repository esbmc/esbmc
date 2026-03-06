f = lambda x: (lambda y: x + y)
g = f(5)
assert g(10) == 15
