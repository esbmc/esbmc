# Negative companion: the populated value must be read back faithfully, so an
# assertion that contradicts it reaches VERIFICATION FAILED.


def main() -> None:
    squares = dict([(i, i * i) for i in [1, 2, 3]])
    assert squares[3] == 8  # squares[3] is 9


main()
