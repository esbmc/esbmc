# Negative companion to github_4807_cmp_unresolved_nondet: under the nondet
# fallback the solver can pick the comparison result freely, so an assertion
# that forces a specific outcome reaches VERIFICATION FAILED instead of the
# old hard abort. This demonstrates the fallback is unconstrained (not
# accidentally folded to a constant).


if __name__ == "__main__":
    a = tuple([1, 2, 3])
    b = tuple([1, 2, 3])
    assert a == b  # nondet bool -- may be false -- must fail
