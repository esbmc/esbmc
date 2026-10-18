d = {"a": 1, 2: 3}
if nondet_bool():
    key = "b"
else:
    key = 2
v = d[key]
