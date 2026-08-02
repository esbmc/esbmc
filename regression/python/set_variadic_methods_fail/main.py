def main() -> None:
    # {1}.union({2}, {3}) contains 3; asserting it does not must fail.
    a = {1}
    u = a.union({2}, {3})
    assert 3 not in u


main()
