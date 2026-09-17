# 6 is the stale pre-append length the literal expansion used to produce.
def main() -> None:
    xs = [1, 2, 3]
    xs.append(4)
    assert len(xs * 2) == 6


main()
