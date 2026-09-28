# Counterpart to github_7872_chained_union.
def foo(y: list[int]) -> int | list[int] | bool:
    return y


M = [1, 2, 3]

assert foo(M) != M
