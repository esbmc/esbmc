# tuple() over an iterable has no element shape the frontend can carry into
# the parameter, so the unpacking is refused rather than typed as a list.


def f(t):
    a, b, c = t
    return a + b + c


assert f(tuple(range(3))) == 3
