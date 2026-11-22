d = {0: 1}
d[0] = 0.5
l = [1]
l.append(d[0])
assert l[1] == 0.5
