# Counterpart to github_7876: accepting every member of the union must not
# make the comparison itself true.
def foo(y: int | list[int]) -> list[int]:
    return y


M = [1, 2, 3]

assert foo(M) != M
