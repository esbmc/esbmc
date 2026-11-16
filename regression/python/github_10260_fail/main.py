# Issue #10260: the loop reads the list's current length on every step.
xs: list[int] = [1, 2, 3]
seen: int = 0
for x in xs:
    xs.pop()
    seen = seen + 1
assert seen == 3
