flag = nondet_bool()
n = nondet_int()
s = "textx"
first = n if flag else "text"
second = n if flag else "text"
if not flag:
    second = s[:4]
    assert first is second
