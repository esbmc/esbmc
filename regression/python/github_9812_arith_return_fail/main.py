# A str on one path and arithmetic on an int parameter on the other (#9812)
def f(flag: bool, n: int):
    if flag:
        return "Hello"
    return n * 3 - 1


x = nondet_int()
flag = nondet_bool()
b = f(flag, x)
if flag:
    assert b == "Hello"
else:
    assert b == x * 3
