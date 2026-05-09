def main() -> None:
    # Basic set comprehension.
    s = {i for i in range(3)}
    assert 0 in s and 1 in s and 2 in s
    assert 3 not in s

    # With a filter.
    odds = {i for i in range(4) if i % 2 == 1}
    assert 1 in odds and 3 in odds
    assert 0 not in odds

    # Expression in the element.
    sq = {i * i for i in range(3)}
    assert 0 in sq and 1 in sq and 4 in sq
main()
