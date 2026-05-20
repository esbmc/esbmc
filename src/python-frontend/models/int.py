# pylint: disable=redefined-builtin,undefined-variable
# This class intentionally shadows the Python built-in `int`: it is the
# operational model ESBMC uses to verify Python programs, so it must
# match the built-in name exactly. The `undefined-variable` disable covers
# the internal `IntWide` annotation used by `bit_length` — see issues
# #1964 / #4642. ESBMC's type_handler maps `IntWide` to a wider signed
# bitvector; it is not a Python symbol, so defining it at module scope
# here would interfere with ESBMC's symbol processing of the OM.
class int:

    @classmethod
    def from_bytes(cls, bytes_data: bytes, big_endian: bool, signed: bool) -> int:
        result: int = 0
        index: int = 0
        step: int = 1
        byte: int = 0
        is_negative: bool = False

        ## If little endian
        if not big_endian:
            index: int = len(bytes_data) - 1
            step: int = -1

        bytes_len: int = len(bytes_data)

        while 0 <= index < bytes_len:
            byte: int = bytes_data[index]
            result: int = (result << 8) + byte
            index: int = index + step

        if signed and bytes_data[-1] & 128 == 128:  # Check MSB of last byte
            is_negative: bool = True

        if signed and is_negative:
            result: int = result - (1 << (8 * len(bytes_data)))

        return result

    @classmethod
    # bit_length() counts the bits needed to represent an integer in binary.
    # `n` is annotated `IntWide` so the OM accepts bignum scalars produced
    # by ** under --ir (e.g. 2**64) without truncating at the call boundary.
    # Narrow int inputs sign-extend on entry. `length` stays `int` — the
    # bit count itself fits trivially. Issues #1964 (Part 2) and #4642.
    def bit_length(cls, n: IntWide) -> int:
        length: int = 0

        while n > 0:
            n: IntWide = n >> 1
            length: int = length + 1  # Count how many times the number is shifted

        return length
