def ks(arr, k):
    for x in arr[k:]:
        yield x


def t(d, e):
    assert list(ks(*d)) == e


t([[2, 1], 1], [1])
