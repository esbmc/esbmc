from collections import Counter

c = Counter()
c.update(a=1)
assert c["a"] == 1
