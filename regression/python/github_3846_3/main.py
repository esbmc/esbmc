# Starred unpacking at the beginning
def f(lst: list[int]):
    *rest, last = lst
    return last


assert f([1, 2, 3]) == 3
assert f([10, 20]) == 20
