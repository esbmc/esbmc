def main() -> None:
    a = {1, 2, 3}
    b = {1, 2}
    assert a.issuperset(b)
    assert not b.issuperset(a)
main()
