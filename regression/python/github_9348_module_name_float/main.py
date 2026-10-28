flag = nondet_bool()
f = nondet_float()
first = f if flag else "text"
second = f if flag else "text"
assert first is second
