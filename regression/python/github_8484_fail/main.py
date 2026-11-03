def g():
    return 42


def call(k):
    return ls[k]()


ls = [g]
a = call(0)
assert a == 41
