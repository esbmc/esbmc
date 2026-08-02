# Pins the mixed-representation arm of the comparison handling (#4807):
# tuple(<list>) is a list object, a tuple literal is a tag-tuple struct.
# An `assert tuple(<list>) == <tuple literal>` is rewritten elementwise by
# the assert-unpack preprocessor pass (and genuinely verifies); outside
# that shape (plain assignment below) the comparison is not yet lowered —
# it falls back to a sound nondet bool. This test checks that conversion
# completes; the nondet result is unconstrained by design.


if __name__ == "__main__":
    # Unpacked by the preprocessor: real elementwise equality, must hold.
    assert tuple([1, 2]) == (1, 2)
    # Not unpacked (assignment, not assert): nondet bool, unasserted.
    cmp_result = (tuple([1, 2]) == (1, 2))
