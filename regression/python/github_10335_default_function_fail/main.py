def text():
    return "s"


def number():
    return 42


def call(f=number):
    return f()


a = call(text)
b = call()
assert a == "s"
assert b == 43
