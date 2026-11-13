from collections import Counter

c = Counter()
c.update([(1, 2)])
assert c[1, 2] == 1
