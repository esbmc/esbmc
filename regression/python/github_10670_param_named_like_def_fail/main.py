def one():
    return 42


def call(call):
    return call()


assert call(one) == 43
