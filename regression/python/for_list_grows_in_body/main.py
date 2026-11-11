xs: list[int] = [1, 2, 3]
seen: int = 0
for x in xs:
    if len(xs) < 5:
        xs.append(0)
    seen = seen + 1
assert seen == 5
assert x == 0
