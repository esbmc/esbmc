def f():
    return 1


def check(a):
    return a == 3


flag = nondet_bool()
assert check(f if flag else 3) == (not flag)
