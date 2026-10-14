# The else clause runs once the comprehension test fails, `break` skips it,
# and `continue` re-evaluates the test.
xs = [1, 5, 6]
n = 0
while [v for v in xs if v > 3]:
    xs.pop()
    n += 1
else:
    n += 10
assert n == 12

ys = [1, 5, 6]
while [v for v in ys if v > 3]:
    if len(ys) == 2:
        break
    ys.pop()
else:
    ys.append(0)
assert ys == [1, 5]

i = 0
k = 0
while [i for _ in range(1)][0] < 4:
    i += 1
    if i % 2 == 0:
        continue
    k += 1
assert k == 2
