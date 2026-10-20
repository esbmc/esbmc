# github #8687: folding a builtin on an int literal above 2**31 must not
# truncate it to 32 bits.
x = float(3000000000)
assert x == 3000000000.0
y = round(3000000000, 1)
assert y == 3000000000.0
z = float(9007199254740993)
assert z == 9007199254740992.0
a = abs(3000000000)
assert a == 3000000000
r = round(3000000000)
assert r == 3000000000
s = str(3000000000)
assert s == "3000000000"
