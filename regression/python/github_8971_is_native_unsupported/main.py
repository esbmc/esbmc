flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
n = 5
assert (x is n) or not (x is n)
