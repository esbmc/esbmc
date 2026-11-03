def annotated(t: tuple[int, list[int]]) -> int:
    t[1].append(5)
    return t[0] + t[1][2]


assert annotated((1, [2, 3])) == 7
