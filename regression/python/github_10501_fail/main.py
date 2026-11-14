from collections import Counter

c = Counter()
c["x"] = 3
for w in ["a", "b", "a"]:
    c[w] += 1
assert c["a"] == 1
