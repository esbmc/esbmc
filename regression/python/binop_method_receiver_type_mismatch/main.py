# A method call whose receiver is a BinOp with mismatched operand types
# (a tagged int subclass mixed with a plain literal). Expects
# VERIFICATION FAILED: bit_length() isn't resolvable through a non-Name
# receiver at all.
class uint64(int):
    pass


def f(index: uint64) -> int:
    position: uint64 = index + 1
    return (position // 256).bit_length()


assert f(uint64(300)) >= 0
