# Three-level generator chain: gen3 -> gen2 -> gen1
def gen1(arr):
    for x in arr:
        yield x


def gen2(arr):
    for y in gen1(arr):
        yield y * 2


def gen3(arr):
    for z in gen2(arr):
        yield z + 1


x = [1, 2, 3]
result = list(gen3(x))
assert result[0] == 3  # (1 * 2) + 1
assert result[1] == 5  # (2 * 2) + 1
