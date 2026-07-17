def main() -> None:
    total: int = 0
    for x in range(4):
        total = total + x
    assert total == 6


main()
