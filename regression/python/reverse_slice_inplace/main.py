# `l[a:b] = reversed(l[a:b])` is an in-place reverse of the sub-range. The
# frontend collapses it to a single in-place range-reverse pass instead of a
# slice-read + reversed() rebuild + slice-assign (three heap copies).

def rev_suffix(xs):
    xs[1:] = reversed(xs[1:])
    return xs

a = [1, 2, 3, 4]
rev_suffix(a)
# suffix [2,3,4] reversed -> [4,3,2]; head 1 unchanged
assert a[0] == 1
assert a[1] == 4
assert a[2] == 3
assert a[3] == 2

# Whole-list reverse via the same idiom.
b = [10, 20, 30]
b[:] = reversed(b[:])
assert b[0] == 30
assert b[1] == 20
assert b[2] == 10

# Bounded interior slice.
c = [5, 6, 7, 8, 9]
c[1:4] = reversed(c[1:4])
assert c[0] == 5
assert c[1] == 8
assert c[2] == 7
assert c[3] == 6
assert c[4] == 9

# Negative bounds exercise the lower+len / upper+len normalization.
f = [1, 2, 3, 4, 5]
f[-3:-1] = reversed(f[-3:-1])  # reverse [3,4] -> [4,3]
assert f[0] == 1
assert f[1] == 2
assert f[2] == 4
assert f[3] == 3
assert f[4] == 5

# Near-miss: RHS is a *different* list, so the idiom must NOT collapse; the
# generic slice-assign path must still produce the correct result.
d = [1, 2, 3]
e = [7, 8]
d[1:] = reversed(e)  # d becomes [1, 8, 7]
assert d[0] == 1
assert d[1] == 8
assert d[2] == 7
