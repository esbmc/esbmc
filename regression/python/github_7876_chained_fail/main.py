# Counterpart to github_7876_chained.
def foo(y: int | list[int] | bool) -> list[int]:
    return y


M = [1, 2, 3]

assert foo(M) != M
