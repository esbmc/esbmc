cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
for e in lst:
    assert e == 1
