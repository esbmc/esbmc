# Operational model for the array module.
# Only the integer typecodes are modelled: an array is a list of ints whose
# every store is range-checked against its typecode, as CPython does.
from typing import Any


def _bounds(typecode: str) -> tuple[int, int]:
    if typecode == "b":
        return (-128, 127)
    if typecode == "B":
        return (0, 255)
    if typecode == "h":
        return (-32768, 32767)
    if typecode == "H":
        return (0, 65535)
    if typecode == "i":
        return (-2147483648, 2147483647)
    if typecode == "I":
        return (0, 4294967295)
    if typecode == "l" or typecode == "q":
        return (-9223372036854775807 - 1, 9223372036854775807)
    if typecode == "L" or typecode == "Q":
        # The upper bound is the model's int range, not 2**64 - 1.
        return (0, 9223372036854775807)
    __ESBMC_assert(typecode != "f" and typecode != "d" and typecode != "u" and typecode != "w",
                   "array.array with a float or unicode typecode is not modelled")
    raise ValueError("bad typecode (must be b, B, u, w, h, H, i, I, l, L, q, Q, f or d)")


class array:

    def __init__(self, typecode: str, initializer: list[int] = None) -> None:
        bounds: tuple[int, int] = _bounds(typecode)
        self.typecode: str = typecode
        self.low: int = bounds[0]
        self.high: int = bounds[1]
        self.items: list[int] = []
        if initializer is not None:
            for v in initializer:
                self.append(v)

    def _check(self, v: int) -> None:
        if v < self.low:
            raise OverflowError("value out of range for array typecode")
        if v > self.high:
            raise OverflowError("value out of range for array typecode")

    def append(self, v: int) -> None:
        self._check(v)
        self.items.append(v)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) != 0

    def __getitem__(self, i: int) -> int:
        return self.items[i]

    def __setitem__(self, i: int, v: int) -> None:
        self._check(v)
        self.items[i] = v

    def tolist(self) -> list[int]:
        return self.items.copy()
