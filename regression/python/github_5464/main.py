# Regression for #5464: an inline list-returning call consumed directly by
# len() -- e.g. len(sorted(d)) -- must bind the call's return value so the
# size is read from the actual sorted list, not an uninitialised temporary.
# Previously these reported a spurious VERIFICATION FAILED (wrong list size).
def main() -> None:
    m = {3: 30, 1: 10, 2: 20}
    assert len(sorted(m)) == 3
    assert len(sorted(m.keys())) == 3
    assert len(sorted(list(m.keys()))) == 3


main()
