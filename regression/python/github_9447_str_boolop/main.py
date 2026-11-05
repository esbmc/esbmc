# github #9447: a str `or`/`and` result is a str, not an untyped value.
r = "a" or "b"
assert r[0] == "a"
assert r < "b"
s = "a"
t = s and "bc"
assert t[0] == "b"
assert t[1] == "c"
e = "" or "xy"
assert e[1] == "y"
c: bool = nondet_bool()
b = c or "a"
if c:
    assert b == True
