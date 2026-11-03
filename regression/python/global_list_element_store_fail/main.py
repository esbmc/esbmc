counter = [0]


def bump() -> int:
    counter[0] = counter[0] + 1
    return counter[0]


bump()
bump()
assert counter[0] == 1
