"""Operational model for the io module."""


class StringIO:
    """io.StringIO: write() overwrites from the current position, as in
    CPython, so StringIO("abc").write("x") leaves "xbc"."""

    def __init__(self, initial_value: str = "") -> None:
        self._buf: str = initial_value
        self._pos: int = 0

    def write(self, s: str) -> int:
        n: int = len(s)
        self._buf = self._buf[:self._pos] + s + self._buf[self._pos + n:]
        self._pos = self._pos + n
        return n

    def read(self) -> str:
        r: str = self._buf[self._pos:]
        self._pos = len(self._buf)
        return r

    def getvalue(self) -> str:
        return self._buf
