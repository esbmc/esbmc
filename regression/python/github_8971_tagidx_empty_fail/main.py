flag = nondet_bool()
x = "" if flag else 5
if isinstance(x, str):
    y = x[0]
