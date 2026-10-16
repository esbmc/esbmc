d = {"a": 1, 2: 3}
c = nondet_bool()
if c:
    key = "b"
else:
    key = 2
try:
    v = d[key]
    assert not c and v == 3
except KeyError:
    assert c
