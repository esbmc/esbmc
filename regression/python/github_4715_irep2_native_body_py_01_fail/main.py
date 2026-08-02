# _fail sibling of github_4715_irep2_native_body_py_01 (W1-loc spike Phase C,
# esbmc/esbmc#4715). Pins the load-bearing half of "byte-identical": under
# --irep2-native-body the body must STILL be converted (via the fallback until
# the native dispatcher is complete), so a genuine assertion violation is
# surfaced as VERIFICATION FAILED and not silently suppressed by skipping
# goto_convert_rec.


def test():
    x = 0

    if x == 0:
        x = 3

    w = 0
    while w < 3:
        w += 1

    assert x == 5, "x is 3, not 5 -- must be detected under --irep2-native-body"
    assert w == 3


test()
