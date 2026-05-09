def main() -> None:
    a = {1, 2}
    b = {1, 2, 3}
    assert a.issubset(b)
    assert not b.issubset(a)
main()
