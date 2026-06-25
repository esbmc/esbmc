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

        # The sign lives in the most-significant byte: index 0 for big-endian,
        # the last byte for little-endian. Index it directly — the model does
        # not support Python negative indexing (bytes_data[-1] read out of
        # bounds), and bytes_data[-1] was also the wrong byte for big-endian.
        sign_index: int = 0 if big_endian else bytes_len - 1
        if signed and bytes_data[sign_index] & 128 == 128:  # MSB of sign byte
            is_negative: bool = True

        if signed and is_negative:
            result: int = result - (1 << (8 * bytes_len))

        return result

    @classmethod
    # bit_length() counts the bits needed to represent an integer in binary.
    # `n` is annotated `IntWide` so the OM accepts bignum scalars produced
    # by ** under --ir (e.g. 2**64) without truncating at the call boundary.
    # Narrow int inputs sign-extend on entry. `length` stays `int` — the
    # bit count itself fits trivially. Issues #1964 (Part 2) and #4642.
    #
    # Implementation: loop bounded by `length`, not by the symbolic input n.
    # The previous `while n > 0: n >>= 1` form forced ESBMC to unwind one
    # symex iteration per bit for any symbolic n, which does not terminate
    # without `--unwind` (issue #4756). Adding `length < 512` to the loop
    # condition gives the unwinder a literal termination bound covering the
    # full --ir bignum IntWide width (512-bit), and narrow callsites still
    # exit at n == 0 well before that — incremental BMC terminates at the
    # first k that proves the property.
    def bit_length(cls, n: IntWide) -> int:
        length: int = 0
        # Soundness: the literal 512 must be >= kPythonBitLengthCap in
        # src/python-frontend/type_handler.cpp, which a static_assert ties
        # to kPythonBignumWidth. The OM is FLAIL-mangled before the C++
        # side is touched, so the literal cannot read the constant — bump
        # both together when widening Python int (#4642). The static_assert
        # turns a silently-wrong bit_length into a build break.
        while length < 512 and n > 0:
            n: IntWide = n >> 1
            length = length + 1
        return length

    @classmethod
    # bit_count() returns the number of ones in the binary representation
    # of the absolute value of the integer (Python 3.10+). `n` is annotated
    # `IntWide` for the same bignum reason as bit_length; narrow inputs
    # sign-extend on entry, so negatives are folded to their magnitude
    # before counting. The loop is bounded by a literal 512-shift counter
    # (matching the --ir bignum IntWide width) rather than by the symbolic
    # input, so the unwinder has a termination bound; narrow callsites exit
    # at n == 0 well before that. The literal 512 shares bit_length's
    # soundness contract (>= kPythonBitLengthCap in type_handler.cpp, tied to
    # kPythonBignumWidth by a static_assert) — bump both together when
    # widening Python int. Issues #1964 / #4642.
    def bit_count(cls, n: IntWide) -> int:
        if n < 0:
            n: IntWide = -n
        count: int = 0
        shifts: int = 0
        while shifts < 512 and n > 0:
            count = count + (n & 1)
            n: IntWide = n >> 1
            shifts = shifts + 1
        return count

    @classmethod
    # conjugate() returns the integer unchanged: the complex conjugate of a
    # real number is itself (int implements the numeric-tower API). Like the
    # other no-argument int methods, the caller's value is passed as `n`.
    def conjugate(cls, n: int) -> int:
        return n
