# Pins the location of a nested ternary under the native body dispatcher
# (W1-loc, esbmc/esbmc#4715). Python's floor division lowers to an arithmetic
# expression carrying an unlocated `if2t` correction term; `if2t` is the one
# value-level IREP2 kind with a location field, so it is the only operand whose
# location survives migrate_expr -- the legacy round-trip stamps it via
# restore_value_locations and the native ASSIGN arm has to do the same.
def isqrt(n: int) -> int:
    x: int = n
    y: int = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


assert isqrt(9) == 3
