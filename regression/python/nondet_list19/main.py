import math

l = []
x = nondet_float()
if not math.isnan(x):
    l.append(x)

assert len(l) >= 0
