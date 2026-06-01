def main() -> None:
    # Negative: any of an all-False tuple is False.
    t = (False, False, False)
    assert any(t)
main()
