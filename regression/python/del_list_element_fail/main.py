# Soundness guard: after `del a[1]` the element values shift, so a[1] is 3, not
# the stale literal 2. ESBMC previously constant-folded the read to the original
# literal and wrongly accepted this; it must now FAIL.
a = [1, 2, 3]
del a[1]
assert a[1] == 2
