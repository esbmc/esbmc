# str.split(None, maxsplit) keeps trailing whitespace in the remainder token.
# Covers both the constant-fold path (literal receiver) and the runtime model
# (parameter receiver), which previously stripped the trailing whitespace and
# diverged from CPython.
from typing import List


def split_runtime(s: str) -> List[str]:
    return s.split(None, 1)


def main() -> None:
    # Constant-fold path.
    a = 'a  b  '.split(None, 1)
    assert len(a) == 2
    assert a[0] == 'a'
    assert a[1] == 'b  '

    b = '  a  '.split(None, 0)
    assert len(b) == 1
    assert b[0] == 'a  '

    # Runtime model path (non-constant receiver).
    c = split_runtime('a  b  ')
    assert len(c) == 2
    assert c[0] == 'a'
    assert c[1] == 'b  '


main()
