def main() -> None:
    # Negative: index 5 is out of range for a 3-element tuple.
    t = (10, 20, 30)
    i = 5
    assert t[i] == 0
main()
