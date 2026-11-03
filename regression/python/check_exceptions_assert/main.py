# AssertionError is not checked, so the assert becomes an assumption, as under
# --python-ignore-assertions, and l[i] stays in bounds.
def main() -> None:
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    x = l[i]
    if i == 1:
        raise AssertionError


main()
