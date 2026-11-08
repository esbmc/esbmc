i = 0
k = 0


def f(x: int) -> int:
    global i
    i = i + 2
    return 0


while i + f(i) < 3:
    k += 1
assert k == 2
