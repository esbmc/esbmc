flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
y = x or 5
assert y or not y
