# pylint: disable=undefined-variable
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


class TextIOWrapper:
    """A text file from open(). Its contents are unknown, so each read returns
    a nondet str (bounded by --nondet-str-length, as input() is)."""

    def __init__(self, file: str, readable: bool, writable: bool) -> None:
        self.name: str = file
        self.readable: bool = readable
        self.writable: bool = writable
        self.closed: bool = False

    def __enter__(self) -> "TextIOWrapper":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.closed = True
        return False

    # CPython raises io.UnsupportedOperation, a subclass of OSError, for a
    # read or write the mode does not allow.
    def read(self) -> str:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if not self.readable:
            raise OSError("not readable")
        return nondet_str()

    def readline(self) -> str:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if not self.readable:
            raise OSError("not readable")
        return nondet_str()

    def write(self, s: str) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if not self.writable:
            raise OSError("not writable")
        return len(s)

    def close(self) -> None:
        self.closed = True


def _open_text(file: str, readable: bool, writable: bool) -> TextIOWrapper:
    """open() in a text mode, which the preprocessor has decoded."""
    if file is None:
        raise TypeError("expected str, bytes or os.PathLike object, not NoneType")
    if nondet_bool():
        raise OSError("cannot open file")
    return TextIOWrapper(file, readable, writable)
