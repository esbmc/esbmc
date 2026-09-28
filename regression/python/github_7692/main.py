def main() -> None:
    assert [(i, i + 1) for i in range(3)][1] == (1, 2)
    assert [(i, i + 1) for i in range(3)][-1] == (2, 3)


main()
