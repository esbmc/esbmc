def pairs(bodies: list[tuple[int, list[int]]]) -> int:
    for (r, v) in bodies:
        return r + v[0]
    return 0


def nested(bodies):
    for (r, (x, y)) in bodies:
        return r + x + y
    return 0


assert pairs([(1, [2, 3])]) == 3
assert nested([(1, [2, 3])]) == 6
