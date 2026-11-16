# Issue #9495: a conditional p = t[i] does not retype p on the other path.
t = (0, ["a", "b"])
p = [1, 2]
c = nondet_bool()
if c:
    p = t[1]
if not c:
    assert p[0] == 2
