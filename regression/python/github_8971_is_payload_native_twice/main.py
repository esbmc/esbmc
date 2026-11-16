flag = nondet_bool()
n = nondet_int()
if flag:
    first = n
else:
    first = "text"
if flag:
    second = n
else:
    second = "text"
assert first is second
