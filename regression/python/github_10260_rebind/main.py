# Issue #10260: rebinding the name leaves the iterator on the original list.
def f(xs: list[int]) -> int:
    n = 0
    for x in xs:
        xs: list[int] = xs + [9, 9]
        n = n + 1
    return n


assert f([1, 2, 3]) == 3
