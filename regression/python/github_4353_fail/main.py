def main() -> None:
    # Negative: width 10 right-aligned makes len 10, not 2.
    s = f"{42:>10}"
    assert len(s) == 2
main()
