# The counterpart of list_range_symbolic_bounds, written so it bites in the
# opposite direction: the broken path made the list empty, so `len(c) == 0`
# held. Materialising the elements makes it false.
def two_args(a: int, b: int) -> None:
    c = list(range(a, b))
    assert len(c) == 0


two_args(1, 10)
