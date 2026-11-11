# github #9960: lists whose recorded elements cannot vouch for a nested list
# keep the element-wise membership check.
def h(n: int) -> list[list[int]]:
    r = []
    for i in range(n):
        r.append([i])
    return r


def f(xs: list[list[int]]) -> bool:
    return [9] in xs


x = h(2)
assert [5] not in x
assert [1] not in [(1, )]
assert [1] not in [[[[[[2]]]]]]
assert not f([[3]])
