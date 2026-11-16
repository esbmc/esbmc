from mgr import acquire


def use(k: int) -> int:
    with acquire(k) as r:
        v = r.get()
    return v


with acquire(1):
    pass

assert use(3) == 3
