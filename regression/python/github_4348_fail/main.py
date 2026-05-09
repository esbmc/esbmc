def main() -> None:
    # Negative: index() on a missing element raises ValueError.
    t = (10, 20, 30)
    _ = t.index(99)
main()
