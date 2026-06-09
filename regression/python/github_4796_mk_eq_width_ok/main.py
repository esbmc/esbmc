# Companion to github_4796_mk_eq_width: a well-typed equality between operands
# of the same bit-width must keep verifying. check_eq_operand_widths only fires
# on a width mismatch, so an ordinary symbolic integer comparison is unaffected.
def check(x: int) -> None:
    assert x == x


check(5)
