# A str on one path and abs() of an int parameter on the other
def f(flag, n):
    if flag:
        return "Hello"
    return abs(n)


x = nondet_int()
flag = nondet_bool()
b = f(flag, x)
if flag:
    assert b == "Hello"
elif x >= 0:
    assert b == x
