def text():
    return "s"


def one():
    return 1


def call(a=one):
    return a()


x = call(text) if nondet_bool() else call()
y = x + 1
