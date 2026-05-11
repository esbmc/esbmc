def main() -> None:
    # Negative: discarding 2 leaves 2 not in s.
    s = {1, 2, 3}
    s.discard(2)
    assert 2 in s
main()
