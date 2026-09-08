# b"\x01\x02"[0] is 1, not 2 (#7550).
def first(b: bytes) -> int:
    return b[0]


def main() -> None:
    assert first(b"\x01\x02") == 2


main()
