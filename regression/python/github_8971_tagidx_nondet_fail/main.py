flag = nondet_bool()
p = nondet_int() if flag else nondet_str()
if isinstance(p, str):
    h = p[0]
