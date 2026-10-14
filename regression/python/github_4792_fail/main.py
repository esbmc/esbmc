# Companion to github_4792: the swap is resolved correctly, so a wrong
# expected value must be caught.
a = [1, 2]
a[0], a[1] = a[1], a[0]
assert a[0] == 1
