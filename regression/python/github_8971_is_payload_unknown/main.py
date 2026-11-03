flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
y = nondet_int() if flag else nondet_str()
same = x is y
assert same or not same
