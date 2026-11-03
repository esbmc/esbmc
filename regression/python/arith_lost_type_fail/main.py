def text():
    return "s"


def one():
    return 1


def call(a):
    return a()


x = call(text) if nondet_bool() else call(one)
y = x + 1
