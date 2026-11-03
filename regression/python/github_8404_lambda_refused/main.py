# One lambda returns a number and the other a str.
def apply(g, v):
    return g(v)


if nondet_bool():
    h = lambda x: x // 2
    r = apply(h, 4)
else:
    h = lambda x: x + "!"
    r = apply(h, "a")
assert r == 2 or r == "a!"
