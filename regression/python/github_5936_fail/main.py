# Negative form of github_5936: before the fix the tuple (1, 5) was read back as
# the int 1, so this assertion "held" and ESBMC reported a false proof. CPython
# raises AssertionError here, and so must ESBMC.
def second(pairs: list[int]) -> int:
    return pairs[0][1]


heap: list = [(1, 5), (1, 2)]
assert second(heap) == 2
