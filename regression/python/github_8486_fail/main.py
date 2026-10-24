d = {"a": 1, 2: 3}
if nondet_bool():
    key = "a"
else:
    key = 2
v = d[key]
assert v == 1
