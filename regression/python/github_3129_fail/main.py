def baz() -> bool:
    return True

def bar() -> int:
    if baz():
        return 42
    else:
        return 0


def foo(i: int) -> None:
    pass

i = bar()
assert i == 0
foo(i)
