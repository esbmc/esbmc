def f(x):
    x = 5
    return x


flag = nondet_bool()
v = nondet_int() if flag else nondet_str()
assert f(v) == 5
