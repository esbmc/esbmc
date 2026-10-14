a = [1, 2, 3]
b = [10, 20, 30]
for x, y in zip(a, b):
    assert y == 10 * x

c = [1, 2, 3]
d = [10, 20]
cnt = 0
for x, y in zip(c, d):
    cnt = cnt + 1
assert cnt == 2
