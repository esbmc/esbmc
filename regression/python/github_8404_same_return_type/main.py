# Both functions return int, so the call through the returned function keeps
# one type and is not refused.
def one():
    return 1


def two():
    return 2


def pick(flag):
    if flag:
        return one
    return two


flag = nondet_bool()
b = pick(flag)()
assert isinstance(b, int)
assert b == (1 if flag else 2)
