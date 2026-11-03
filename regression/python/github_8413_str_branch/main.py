def sign(a: int) -> str:
    if a > 0:
        s = "P"
    elif a < 0:
        s = "N"
    else:
        s = "Z"
    return s


a = nondet_int()
if a > 0:
    s = "P"
else:
    s = "N"
assert a <= 0 or s == "P"
assert a > 0 or s == "N"
assert a <= 0 or s != "N"
assert a > 0 or s < "P"
assert a <= 0 or "P" in s
assert a > 0 or "P" not in s
assert a <= 0 or f"<{s}>" == "<P>"
assert a > 0 or s + "x" == "Nx"
assert a <= 0 or "-".join([s, s]) == "P-P"
assert a != 0 or sign(a) == "Z"
assert a >= 0 or sign(a) == "N"

w = "pos"
assert w == "pos"
if a > 0:
    w = "pos"
else:
    w = "negative"
assert a <= 0 or w == "pos"
assert a > 0 or w == "negative"
assert a <= 0 or len(w) == 3
assert a > 0 or w.startswith("neg")

t = "N"
if a > 0:
    t += "x"
assert a <= 0 or t == "Nx"
assert a > 0 or t == "N"

if a > 0:
    u = "ab"
else:
    u = "x"
assert a <= 0 or u == "ab"
u = "longer"
assert u == "longer"

v = "N"
i = 0
while i < 2:
    if i == 1:
        assert a <= 0 or v == "P"
    if a > 0:
        v = "P"
    i += 1
assert a > 0 or v == "N"

try:
    if a > 0:
        raise ValueError("positive")
    r = "N"
except ValueError:
    r = "P"
assert a <= 0 or r == "P"
assert a > 0 or r == "N"

q = "N"
j = 0
while q == "N" and j < 3:
    q = "P"
    j += 1
assert j == 1
