def main() -> None:
    d = {1: 10, 2: 20}
    e = d.copy()
    # Negative: the copied dict's value at key 1 is 10, not 99.
    assert e[1] == 99
main()
