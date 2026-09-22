# Counterpart to github_7872_bare_list: declining to widen must not make every
# comparison true.
def foo(y: list) -> int | list:
    return y


M = [1, 2, 3]

assert foo(M) != M
