l = [10, 20, 30, 40, 50]
t = l[None:None:-1]
assert t[0] == 50
r = l[:None]
assert len(r) == 4
