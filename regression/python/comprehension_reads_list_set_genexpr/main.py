n = 0


def g() -> int:
    global n
    n += 1
    return n


xs = [g(), g()]
s = {b for b in xs}
t = sum(b for b in xs)
u = any(b > 1 for b in xs)
assert len(s) == 2
assert t == 3
assert u
assert n == 2
