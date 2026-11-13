d = {'a': 1}
v = d.get('b', 0)
xs = [3, 1, 2]
xs.sort()
for k in d:
    v += 1
assert v == 1 and xs == [1, 2, 3]
