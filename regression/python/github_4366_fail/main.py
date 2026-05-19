def main() -> None:
    # Negative: a is a strict subset of b, so b is NOT a subset of a.
    a = {1, 2}
    b = {1, 2, 3}
    assert b.issubset(a)
main()
