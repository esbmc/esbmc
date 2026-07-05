# Issue #5185 (negative variant): the subscript must produce a real verdict, not
# an internal abort. Here the asserted element is wrong, so verification must
# FAIL rather than crash with "Unrecognized address_of operand".
def main() -> None:
    assert ("a", ".", "b.c")[2] == "wrong"


main()
