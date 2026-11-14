xs: list[int] = [1, 2, 3]
for x in xs:
    if x == 2:
        xs.append(9)
    assert x != 9
