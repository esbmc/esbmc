d = {"a": 1, 2: 3}
c = nondet_bool()
if c:
    key = "a"
else:
    key = 2
v = d[key]
assert v == (1 if c else 3)

key2 = "a" if nondet_bool() else 2
w = d[key2]
assert w == 1 or w == 3
