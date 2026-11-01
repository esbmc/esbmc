# Issue #9495: a list read from a tuple by index keeps its float elements.
t = (0, [1.0, 1.0])
p = t[1]
vx, vy = p
assert vx + vy != 2.0
