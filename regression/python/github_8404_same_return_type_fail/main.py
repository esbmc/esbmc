# Both functions return int; the assertion is wrong when flag is false.
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
assert b == 1
