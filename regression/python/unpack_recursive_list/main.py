def head(arr: list[int], d: int) -> int:
    first, second = arr
    if d:
        head([5, 6], d - 1)
    return first


def star_head(arr: list[int], d: int) -> int:
    first, *rest = arr
    if d:
        star_head(rest, d - 1)
    return first


def powerset(arr):
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return rest_subsets + [[first] + subset for subset in rest_subsets]
    return [[]]


assert head([1, 2], 1) == 1
assert star_head([1, 2, 3], 2) == 1
r = powerset(["a", "b"])
assert r[3][0] == "a"
assert r[3][1] == "b"
