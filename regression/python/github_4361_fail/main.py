def main() -> None:
    # Negative: min by p[1] selects "a", not "b".
    pairs = [(1, "b"), (3, "a"), (2, "c")]
    assert min(pairs, key=lambda p: p[1])[1] == "b"
main()
