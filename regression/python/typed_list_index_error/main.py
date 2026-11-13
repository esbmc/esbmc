l = [3, 1, 2]
l[1] = 7
assert l[-1] == 2
assert l[1] == 7
l[0] += 4
assert l[0] == 7
caught = False
try:
    x = l[3]
except IndexError:
    caught = True
assert caught
caught = False
try:
    l[-4] = 0
except IndexError:
    caught = True
assert caught
assert len(l) == 3
