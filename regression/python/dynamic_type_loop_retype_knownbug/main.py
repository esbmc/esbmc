# x is reassigned int/str inside a loop body. ESBMC silently corrupts the
# string value instead of tracking the type, so the assertion is currently
# VERIFICATION FAILED (a false positive; the assertion always holds).

x = 0
i = 0
while i < 3:
    if nondet_bool():
        x = 1
    else:
        x = "a"
    i += 1
assert x == 1 or x == "a"
