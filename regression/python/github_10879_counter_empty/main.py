from collections import Counter

c = Counter()
c[1] = c[1] + 1
assert c[1] == 1
assert c[2] == 0
