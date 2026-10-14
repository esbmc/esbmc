# The argument's type depends on pair's own parameter, which the same pass
# types. Typing `pairs` before that left it list[tuple] with an unknown
# element shape, and the nested unpacking had nothing to read.


def pair(l):
    result = []
    result.append((l[0], l[1]))
    return result


def first(pairs):
    for ((a, b), (c, d)) in pairs:
        return a + c
    return 0


assert first(pair([(1, 2), (3, 4)])) == 4
