# Regression for #4807: relational ops over operands of unresolved type
# (e.g. the result of an undispatched `tuple(<list>)` call) used to abort
# GOTO conversion with "Unsupported comparison with unresolved operand
# type". The handler now returns a sound nondet bool fallback, so GOTO
# conversion succeeds and verification proceeds against the symbolic
# comparison result.
#
# This test exercises the conversion path itself -- evaluating the
# comparison must no longer abort. The nondet bool value is intentionally
# not asserted on (its outcome is unconstrained); a separate _fail test
# pins the unconstrained behaviour.


if __name__ == "__main__":
    a = tuple([1, 2, 3])
    b = tuple([1, 2, 3])
    # Previously aborted; now lowers to a nondet bool. Bind to a name so
    # the comparison expression is actually emitted in GOTO.
    cmp_result = (a == b)
    # No assertion on cmp_result: this test only checks that conversion
    # completed and the program has no other VCCs that would fail.
