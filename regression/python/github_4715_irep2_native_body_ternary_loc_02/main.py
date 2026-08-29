# Legacy-path control for github_4715_irep2_native_body_ternary_loc_01: the same
# body under --no-irep2-native-body, which is where the ternary's location comes
# from in the first place (restore_value_locations, then migrate_expr). It passes
# before the fix as well as after, and exists so the pair pins both halves of the
# A/B rather than only the native one. Keep it line-for-line aligned with _01 --
# both test.desc regexes name the same source line.
def isqrt(n: int) -> int:
    x: int = n
    y: int = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


assert isqrt(9) == 3
