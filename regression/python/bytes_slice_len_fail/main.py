def main() -> None:
    # len(bytes([1, 2, 3, 4])[1:3]) is 2, not 3. (Pre-fix it wrongly computed 1
    # via strlen over the wide-int byte representation.)
    b = bytes([1, 2, 3, 4])
    assert len(b[1:3]) == 3


main()
