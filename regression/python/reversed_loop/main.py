xs = [1, 2, 3]
out = []
for x in reversed(xs):
    out.append(x)
assert out[0] == 3 and out[1] == 2 and out[2] == 1
