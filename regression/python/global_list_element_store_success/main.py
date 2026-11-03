counter = [0]
totals = [0, 0]


def bump() -> int:
    counter[0] = counter[0] + 1
    return counter[0]


def add(n: int) -> None:
    global totals
    for i in range(n):
        totals[1] += 2


def shadow() -> int:
    counter = [5]
    counter[0] = 7
    return counter[0]


assert bump() == 1
assert bump() == 2
add(3)
assert totals[1] == 6
assert shadow() == 7
assert counter[0] == 2
