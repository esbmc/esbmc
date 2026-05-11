def main() -> None:
    s = {1, 2}
    s.update({3})
    assert 3 in s and 1 in s
main()
