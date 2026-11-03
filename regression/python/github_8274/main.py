k: int = 3


def step() -> float:
    global k
    k -= 1
    return float(k)


n: int = 0
while int(step()) > 0:
    n += 1
assert n == 2
assert k == 0
