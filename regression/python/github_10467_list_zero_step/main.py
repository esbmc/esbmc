l = [10, 20, 30, 40, 50]
s = 0
caught = 0
try:
    r = l[::s]
except ValueError:
    caught = 1
assert caught == 1
