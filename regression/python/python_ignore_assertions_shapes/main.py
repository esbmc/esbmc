# --python-ignore-assertions drops `assert lst` and `assert f()` like a plain
# assert: each only ends the path.
def g() -> None:
    pass


def main():
    l: list[int] = []
    if nondet_bool():
        assert l
    else:
        assert g()


main()
