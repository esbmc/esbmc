# Issue #5185: subscripting a tuple literal with heterogeneous string elements
# at a non-zero index aborted SMT encoding ("Unrecognized address_of operand")
# because the element member of the inline constant struct rvalue is not
# addressable. Each access here matches the CPython result.
def main() -> None:
    assert ("a", ".", "b.c")[0] == "a"
    assert ("a", ".", "b.c")[1] == "."
    assert ("a", ".", "b.c")[2] == "b.c"
    assert ("a", ".", "b.c")[-1] == "b.c"
    # slice over the literal then index the longer element
    assert ("a", ".", "b.c")[1:3][1] == "b.c"


main()
