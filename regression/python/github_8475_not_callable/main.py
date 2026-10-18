def f():
    return 1


def call(a):
    return a()


c = call(f if nondet_bool() else 3)
