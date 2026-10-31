flag = nondet_bool()
n = nondet_int()
first = n if flag else "a b"
second = n if flag else "a b"
if not flag:
    assert first is second
