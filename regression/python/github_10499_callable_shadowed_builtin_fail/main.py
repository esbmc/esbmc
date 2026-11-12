# github #10499: a parameter named int does not make the builtin not callable
def f(int):
    return int


f(1)
assert not callable(int)
