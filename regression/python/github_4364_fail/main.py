def main() -> None:
    # Negative: sort by p[1] selects "a" first, not "b".
    pairs = [(1, "b"), (3, "a"), (2, "c")]
    pairs.sort(key=lambda p: p[1])
    assert pairs[0][1] == "b"
main()
