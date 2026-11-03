def first_plus_head(t):
    return t[0] + t[1][0]


def annotated(t: tuple[int, list[int]]) -> int:
    t[1].append(5)
    return t[0] + t[1][2]


def with_dict(t: tuple[int, dict[str, int]]) -> int:
    return t[0] + t[1]["a"]


assert first_plus_head((1, [2, 3])) == 3
xs = [2, 3]
assert annotated((1, xs)) == 6
assert len(xs) == 3
assert with_dict((1, {"a": 4})) == 5
