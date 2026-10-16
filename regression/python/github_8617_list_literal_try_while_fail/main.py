a = nondet_int()
m = [0]
i = 0
while i < 1:
    if a > 0:
        m = [1]
        break
    i += 1
else:
    m = [2]
assert m[0] == 2
