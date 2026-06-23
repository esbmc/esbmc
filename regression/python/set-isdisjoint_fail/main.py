def main() -> None:
    a = {1, 2, 3}
    b = {3, 4, 5}
    # 3 is shared, so the sets are NOT disjoint; this assertion must fail.
    assert a.isdisjoint(b)
main()
