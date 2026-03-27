def gen1(arr):
    for x in arr:
        yield x


def gen2(arr):
    for y in gen1(arr):
        yield y


x = [1, 4, 6]
y = list(gen2(x))
assert y[0] == 1
