
def flatten(arr):
    for x in arr:
        if isinstance(x, list):
            for y in flatten(x):
                yield y
        else:
            yield x


assert list(flatten([[[1, [], [2, 3]], [[4]], 5]])) == [1, 2, 3, 4, 5]
assert list(flatten([1, 2, 3, [[4]]])) == [1, 2, 3, 4]
assert list(flatten([1, 4, 6])) == [1, 4, 6]
assert list(flatten([["moe", "curly", "larry"]])) == ["moe", "curly", "larry"]
assert list(flatten(["a", "b", ["c"], ["d"], [["e"]]])) == ["a", "b", "c", "d", "e"]