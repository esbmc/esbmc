import helper


def func():
    return 42


def pick(c):
    if c:
        f = helper.func
    else:
        f = func
    return f()


assert pick(False) == 42
assert pick(True) == 42
