def text():
    return "s"


def number():
    return 42


def call(f):
    return f()


a = call(f=text)
b = call(f=number)
assert a == "s"
assert b == 42
