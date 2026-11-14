n = nondet_int()
l = [0] * n
l.extend(l)
d = 5
total = 0
for x in l:
    total = total + 10 // d
