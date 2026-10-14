# try/except/else/finally is now supported: with no exception the else clause
# runs after the body, then the finally runs on the way out. (Valid Python;
# runs cleanly under CPython.)
x = 0
try:
    x = 1
except ValueError:
    x = 2
else:
    x = 3
finally:
    x = 4
assert x == 4
