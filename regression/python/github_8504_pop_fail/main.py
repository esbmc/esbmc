s = []
if nondet_bool():
    s.append(1)
try:
    x = s.pop()
except KeyError:
    x = 0
