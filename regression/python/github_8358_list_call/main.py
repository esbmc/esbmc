# list() over a range gives the parameter list[int] rather than a bare list
# whose untyped elements the solver cannot encode.


def f(t):
    a, b, c = t
    return a + b + c


assert f(list(range(3))) == 3
