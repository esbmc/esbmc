n: int = nondet_int()

if n > 0:
    s = "abab"
else:
    s = "cdcd"

t = s.replace("ab", "xy")
assert len(t) == 4
