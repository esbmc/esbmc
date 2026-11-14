from collections import Counter

c = Counter()
c[1, 2] = 3
c.clear()
assert c[1, 2] == 3
