from collections import Counter

c = Counter()
c[1, 2] = 3
c[4, 5] = 6
c.update()
assert c[1, 2] == 3
c.clear()
assert not c
assert c[1, 2] == 0
c[4, 5] = 1
assert c[4, 5] == 1
