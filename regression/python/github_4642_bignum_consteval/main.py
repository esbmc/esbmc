def f(n):
    # n is unused so consteval would fold f(anything) to 0. With the bignum
    # tag bypassing the const-args pre-scan, the call falls back to the
    # normal builder, whose recursion into get_literal raises the overflow
    # diagnostic — instead of silently swallowing the bignum literal.
    return 0


y = f(18446744073709551616)
assert y == 0
