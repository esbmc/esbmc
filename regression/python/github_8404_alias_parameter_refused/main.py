# call is invoked through the alias c, which passes f on to its parameter g,
# and f returns str or int.
def s():
    return "a"


def i():
    return 1


def call(g):
    return g()


c = call
if nondet_bool():
    f = s
else:
    f = i
r = c(f)
if f is i:
    assert isinstance(r, int)
