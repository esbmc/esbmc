def expect_int(value: int) -> int:
    return value + 1


def main() -> None:
    number: int = "forty-two"
    expect_int("fail")


main()
