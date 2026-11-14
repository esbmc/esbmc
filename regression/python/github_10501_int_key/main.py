from collections import Counter

c = Counter()
c[7] = 2
c[7] += 1
assert c[7] == 3
assert c[8] == 0
