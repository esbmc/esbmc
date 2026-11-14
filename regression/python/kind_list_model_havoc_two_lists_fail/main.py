n = nondet_int()
m = nondet_int()
a = [1] * n
b = [0] * m
total = 0
for x in a:
    total = total + 10 // len(b)
