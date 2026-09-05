# A keyword moved into a positional slot stayed in node.keywords too, so the
# node claimed the same parameter twice. A range loop visits its body a second
# time, and the duplicate check then rejected the frontend's own output (#7542).
def rslice(n: int, allow_empty: bool = False) -> int:
    return n if allow_empty else 0


def main() -> None:
    total = 0
    for _ in range(2):
        total += rslice(3, allow_empty=True)
    assert total == 6

    once = 0
    for _ in range(1):
        once += rslice(3, allow_empty=True)
    assert once == 3

    assert rslice(3, allow_empty=True) == 3
    assert rslice(3) == 0


main()
