flag = nondet_bool()
m = nondet_int()
n = nondet_int()
first = m if flag else "text"
second = n if flag else "text"
if flag:
    assert first is second
