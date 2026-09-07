# Positive twin of nondet_list24_fail: the same call-argument position, but a
# property that holds however the elements are chosen. Pins that expanding the
# call keeps the length contract, not just that the elements differ.
def check(z: list[int]) -> None:
    assert len(z) <= 3


check(nondet_list(3))
