# A zero step raises ValueError.
s = nondet_int()
total = 0
for k in range(0, 3, s):
    total = total + k
print(total)
