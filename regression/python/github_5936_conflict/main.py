# The frontend emits one GOTO function per Python function, so a parameter fed
# tuples of different element types by different call sites has no single
# correct element type: it must be left on its annotation rather than bound to
# one caller's type (GitHub #5936). Neither call reads an element, so both are
# still verifiable.
def size(pairs: list[int]) -> int:
    return len(pairs)


ints: list = [(1, 5), (1, 2)]
strs: list = [("a", "b")]
assert size(ints) == 2
assert size(strs) == 1
