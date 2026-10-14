# Typing a tuple() argument as a list would make this isinstance check fail
# where CPython passes it. The parameter must stay refused instead.


def f(t):
    assert isinstance(t, tuple)
    a, b, c = t
    return a + b + c


assert f(tuple(range(3))) == 3
