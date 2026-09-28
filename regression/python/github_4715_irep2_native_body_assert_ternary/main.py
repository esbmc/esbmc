# remove_sideeffects is entered for a top-level ternary even when it has no side
# effect, and under --validate-violation-witness it lowers `c ? a : b` to
# DECL/IF/GOTO so the `?` column reaches the branching waypoint. The assert and
# assume arms gated on has_sideeffect alone and emitted a single ASSERT here,
# where every sibling arm mirrors remove_sideeffects' full entry condition.
def f(c: bool, a: bool, b: bool) -> None:
    assert a if c else b


f(True, True, False)
