n = 0


def g() -> int:
    global n
    n += 1
    return n


xs = [g(), g()]
total = 0
for b in xs:
    total += b
assert total == 3
assert n == 2
