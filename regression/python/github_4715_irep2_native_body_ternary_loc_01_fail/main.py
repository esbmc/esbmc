# Negative half of github_4715_irep2_native_body_ternary_loc_01: stamping the
# ternary's location must not change the verdict on the same body.
def isqrt(n: int) -> int:
    x: int = n
    y: int = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


assert isqrt(9) == 4
