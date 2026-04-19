from collections import Counter

c = Counter()
assert c[2, 3] == 0
c[2, 3] = 11
assert c[2, 3] == 11
