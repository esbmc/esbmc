class A:
    pass


flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
assert x is A
