SIZE: int = 3


def last_square(values: list[int]) -> int:
    return values[SIZE]


def main() -> None:
    values: list[int] = [0, 1, 4, 9]
    print(last_square(values))


main()
