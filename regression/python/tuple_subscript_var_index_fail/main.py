# Negative variant: t[0] is 10, so asserting it equals 99 must fail.
def get(t: tuple[int, int, int], i: int) -> int:
    return t[i]


def main() -> None:
    t = (10, 20, 30)
    assert get(t, 0) == 99


main()
