# `get()(arg())` runs `get()` once, before the argument (#8474).
order: int = 0
get_at: int = 0
arg_at: int = 0


def inc(n: int) -> int:
    return n + 1


def get():
    global order, get_at
    order = order + 1
    get_at = order
    return inc


def arg() -> int:
    global order, arg_at
    order = order + 1
    arg_at = order
    return 1


b = get()(arg())
assert b == 2
assert get_at == 1
assert arg_at == 2
assert order == 2
