def main() -> None:
    xs: list[int] = [3, 1, 2]
    xs.append(4)
    assert len(xs) == 4
    assert max(xs) == 4


main()
