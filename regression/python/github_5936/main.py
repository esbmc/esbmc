# A list[int]-annotated parameter fed a list of tuples must model the tuples,
# not reinterpret each tuple's payload as an int (GitHub #5936). Python does
# not enforce annotations, so the caller's element type wins.
def second(pairs: list[int]) -> int:
    return pairs[0][1]


heap: list = [(1, 5), (1, 2)]
assert second(heap) == 5
