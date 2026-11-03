flag = nondet_bool()
n = nondet_int()
first = n if flag else "text"
second = n if flag else "text"
if not flag:
    second = "".join(["te", "xt"])
    assert first is second
