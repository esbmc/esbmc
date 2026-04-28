def flatten(arr):
    for x in arr:
        if isinstance(x, list):
            for y in flatten(x):
                yield y
        else:
            yield x


# Flat list: no recursion needed
result = list(flatten([1]))
assert len(result) == 1
