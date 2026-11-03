l = [1]


def bump() -> list[int]:
    global l
    l = [2]
    return [3]


r = l + bump()
assert r[0] == 2
