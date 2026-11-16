# github #10120: a nested and/or in a while test short-circuits.
xs: list[int] = []
n = 0
while (not xs or xs[0] > 0) and n < 1:
    n += 1
assert n == 1
