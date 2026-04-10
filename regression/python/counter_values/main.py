from collections import Counter

c = Counter()
c[2, 3] = 11
c[4, 5] = 7
assert max(c.values()) == 11
