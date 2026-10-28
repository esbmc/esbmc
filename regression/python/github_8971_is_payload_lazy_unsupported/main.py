flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
y = nondet_int() if flag else nondet_str()
assert (x is y) or not (x is y)
