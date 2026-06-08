def main() -> None:
    # set.add inserts when absent.
    s = {1, 2}
    s.add(3)
    assert 3 in s
    assert 1 in s and 2 in s

    # set.add is idempotent for present elements.
    s.add(3)
    assert 3 in s

    # set.discard removes when present.
    s.discard(2)
    assert 2 not in s
    assert 1 in s and 3 in s

    # set.discard is silent when the element is absent.
    s.discard(99)
    assert 1 in s and 3 in s
main()
