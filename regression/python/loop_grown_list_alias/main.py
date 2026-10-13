xs = []
c = 0
for n in range(2):
    for p in xs:
        c += 1
    ys = xs
    ys.append(n)
assert c == 1
