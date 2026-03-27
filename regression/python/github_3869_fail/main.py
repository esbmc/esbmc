def f(arr):
    for x in arr:
        for y in f([]):
            yield y
        yield x


assert list(f([1, 4, 6])) == [1, 4, 7]
