# Negative companion to github_4807_splitlines_nondet: confirms the
# nondet fallback's empty-list result reaches the solver -- with `s`
# bound to a constant string the original constant-folded path returns
# a list of length >= 1, so an assertion forcing len > 99 must fail.


def split_lines(s: str):
    return s.splitlines()


if __name__ == "__main__":
    # The receiver here is the literal "a\nb" so this resolves via the
    # original constant path (not the nondet fallback) -- the test is
    # pinning that the comparison reaches the solver and FAILS, not that
    # the fallback itself is invoked.
    parts = "a\nb".splitlines()
    assert len(parts) > 99  # must fail: actual length is 2
