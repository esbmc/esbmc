from collections import Counter

c = Counter()
top = c.most_common(1)
# c is empty so top is an empty list
assert len(top) == 0
