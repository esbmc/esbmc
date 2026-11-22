# Issue #10260: a list that grows in the body is iterated to its new end.
xs = [1, 2, 3]
seen = 0
for x in xs:
    if len(xs) < 5:
        xs.append(0)
    seen = seen + 1
assert seen == 3
