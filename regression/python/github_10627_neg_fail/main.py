# A str on one path and a negated int parameter on the other
def f(flag, n):
    if flag:
        return "Hello"
    return -n


x = nondet_int()
flag = nondet_bool()
b = f(flag, x)
if flag:
    assert b == "Hello"
else:
    assert b == x
