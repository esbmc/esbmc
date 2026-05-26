# Regression for the segfault when subscripting/slicing the result of a
# base-conversion builtin (bin/hex/oct) applied to a non-constant integer.
# bin(N) with symbolic N yields a propagated exception expression; slicing it
# previously built an address_of over a non-array operand and crashed the
# converter during GOTO generation. The function is never called, so the body
# is unreachable and verification succeeds once conversion no longer crashes.


def solve(N):
    return bin(N)[2:]


if __name__ == "__main__":
    assert bin(5)[2:] == "101"
