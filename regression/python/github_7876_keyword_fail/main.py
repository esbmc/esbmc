# Counterpart to github_7876_keyword.
def foo(y: int | list[int]) -> list[int]:
    return y


M = [1, 2, 3]

assert foo(y=M) != M
