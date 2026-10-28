flag = nondet_bool()
x = "ab" if flag else 5
if isinstance(x, str):
    y = x[2]
