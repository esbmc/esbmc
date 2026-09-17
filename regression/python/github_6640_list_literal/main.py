def inc(x: int) -> int:
    return x + 1


def main() -> None:
    fs = [inc]
    assert len(fs) == 1


main()
