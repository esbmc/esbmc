n = 0


def bump(v: int) -> bool:
    global n
    n += 1
    return v > 0


xs = [1]
ys = [1]
i = len(xs) - 1
r = xs and bump(ys[i])
assert n == 1
