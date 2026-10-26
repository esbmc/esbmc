xs = []
c = 0
for n in range(2):
    for p in xs:
        c += 1
    xs.append(n)
assert c == 0
