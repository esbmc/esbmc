# github #10120: the guard of xs[0] runs on each iteration that reads it.
xs: list[int] = []
n = 0
while (n > 0 and xs[0] > 0) or n < 1:
    n += 1
assert n == 1
