# #10456: a zero range() step raises ValueError, which is outside the
# AssertionError family checked here.
s = nondet_int()
total = 0
for k in range(0, 3, s):
    total = total + k
print(total)
