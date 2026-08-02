# Negative companion to github_4807_cmp_unresolved_nondet: tuples built
# from lists with different contents compare unequal, so the assertion
# fails. (This test originally pinned the unconstrained nondet-bool
# fallback for unresolved operand types; tuple() over a list now resolves
# to a real elementwise comparison.)


if __name__ == "__main__":
    a = tuple([1, 2, 3])
    b = tuple([1, 2, 4])
    assert a == b  # contents differ -- must fail
