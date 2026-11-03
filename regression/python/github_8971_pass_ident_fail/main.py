def ident(x):
    return x


flag = nondet_bool()
v = nondet_int() if flag else nondet_str()
r = ident(v)
assert r != v
