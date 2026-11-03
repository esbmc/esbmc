flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
assert (not x) == ((isinstance(x, int) and x != 0) or (isinstance(x, str) and x != ""))
