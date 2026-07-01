# Negative: after reversing the suffix in place, the last element is 2, not 4.
# ESBMC must report the assertion violation (the idiom must not silently no-op).

def rev_suffix(xs):
    xs[1:] = reversed(xs[1:])
    return xs

a = [1, 2, 3, 4]
rev_suffix(a)
assert a[3] == 4  # wrong: a[3] is 2 after the in-place reverse
