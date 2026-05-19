def main() -> None:
    # Negative: a + b has 4 elements, so c[3] is 4, not 99.
    a = (1, 2)
    b = (3, 4)
    c = a + b
    assert c[3] == 99
main()
