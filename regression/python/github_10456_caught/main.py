# #10456: the zero-step ValueError can be caught.
s = nondet_int()
caught = False
try:
    for k in range(0, 3, s):
        pass
except ValueError:
    caught = True
assert caught == (s == 0)
