flag = nondet_bool()
n = nondet_int()
first = n if flag else "text"
second = n if flag else "text"
if flag:
    assert first is not second
