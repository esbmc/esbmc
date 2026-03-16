# Starred unpacking in the middle
def f(lst: list[int]):
    first, *middle, last = lst
    return first + last

assert f([1, 2, 3, 4]) == 5
assert f([10, 20, 30]) == 40
