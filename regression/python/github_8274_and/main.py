k: int = 0


def step() -> float:
    global k
    k += 1
    return float("nan")


n: int = 0
go: bool = False
while go and int(step()) > 0:
    n += 1
assert n == 0
assert k == 0
