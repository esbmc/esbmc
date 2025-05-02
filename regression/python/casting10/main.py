# Float vs int → allowed, should pass
f = float(3)
assert f < 4
f = 5.0
g = float(5)
assert f >= 2
assert g >= 2

# Float vs float → allowed
f = float(2.0)
assert f == 2.0

# String vs string → allowed
assert "A" < "B"
