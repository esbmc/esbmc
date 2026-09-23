# An annotation contradicting its literal does not type the parameter (#7745).
x: int = 2.5
g = lambda n: n + 0
assert g(x) == 2
