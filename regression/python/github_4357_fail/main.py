from collections import deque


def main() -> None:
    # Negative: a fresh deque has length 1 after one append, not 99.
    d = deque()
    d.append(1)
    assert len(d) == 99
main()
