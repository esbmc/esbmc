from mod import greet

cond = nondet_bool()
x = nondet_str() if cond else nondet_int()
if cond:
    assert greet(x) != ""
