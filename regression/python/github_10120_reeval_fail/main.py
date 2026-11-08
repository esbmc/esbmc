ys = [0, 5]
n = 0
while (not ys or ys[-1] > 0) and n < 3:
    ys.pop()
    n += 1
assert n == 2
