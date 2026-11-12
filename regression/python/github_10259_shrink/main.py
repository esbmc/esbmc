xs = [1, 2, 3]
seen = []
for x in xs:
    xs.pop()
    seen.append(x)
assert seen == [1, 2]
