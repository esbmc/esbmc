# Inside a function, `get()()` calls the function `get()` returns rather
# than the enclosing function (#8474).
def f():
    return 1


def get():
    return f


def main() -> None:
    b = get()()
    assert b == 1


main()
