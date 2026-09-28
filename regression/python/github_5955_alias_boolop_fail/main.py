# `l or [9]` yields l itself, so m aliases l and the append invalidates the
# seed; the assert must not fold to a proof (GitHub #5955 review F1).
l = [1, 2]
m = l or [9]
m.append(3)
assert l.count(3) == 0
