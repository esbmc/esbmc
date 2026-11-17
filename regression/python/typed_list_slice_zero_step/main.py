l = [1, 2, 3]
caught = False
try:
    r = l[::0]
except ValueError:
    caught = True
assert caught
assert len(l[::1]) == 3
