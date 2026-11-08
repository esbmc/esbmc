xs: list[int] = []
n = 0
while (not xs or xs[0] > 0) and n < 1:
    n += 1
assert n == 1

n = 0
while n < 1 and (not xs or xs[0] > 0):
    n += 1
assert n == 1

n = 0
while (len(xs) > 0 and xs[0] > 0) or n < 1:
    n += 1
assert n == 1

n = 0
while n < 1 and not (len(xs) > 0 and xs[0] > 0):
    n += 1
assert n == 1

ys = [0, 5]
n = 0
while (not ys or ys[-1] > 0) and n < 3:
    ys.pop()
    n += 1
assert n == 1
