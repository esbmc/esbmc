xs = [1]
c = 0
i = 0
while i < 2:
    for p in xs:
        c += p
    xs.append(5)
    i += 1
assert c == 7
