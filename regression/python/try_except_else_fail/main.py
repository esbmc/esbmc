# With no exception the else clause runs (x==3), so x==2 must not hold.
x = 0
try:
    x = 1
except:
    x = 2
else:
    x = 3
assert x == 2
