def last() -> float:
    v = None
    v = 3
    for v in [1.5, 2.5]:
        pass
    return v + 1


assert last() == 3.5
