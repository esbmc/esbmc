flag = nondet_bool()
v = "ab" if flag else 5
r = v.startswith("a")
