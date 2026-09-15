# The keyword must still reach its parameter after the rewrite: with
# allow_empty=True the call is 3, not 0 (#7542).
def rslice(n: int, allow_empty: bool = False) -> int:
    return n if allow_empty else 0


def main() -> None:
    total = 0
    for _ in range(2):
        total += rslice(3, allow_empty=True)
    assert total == 0


main()
