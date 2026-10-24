# A tuple() argument reaches the parameter as the list the model builds,
# so the unpacking reads it (#8777).


def f(t):
    a, b, c = t
    return a + b + c


assert f(tuple(range(3))) == 3
