# inner() writes x by name, so the return must not put back x's old value.
def outer() -> int:
    x = 1
    def inner() -> None:
        nonlocal x
        x = 5
    inner()
    assert x == 1
    return x
outer()
