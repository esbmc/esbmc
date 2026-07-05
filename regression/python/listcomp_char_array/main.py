# List comprehension over a char-array variable (not a string literal) drives
# the array-length branch in list_comprehension.cpp whose `i < length` loop
# condition is built in IREP2 with gen_typecast_arithmetic width reconciliation.
def main() -> None:
    s = "abc"
    chars = [c for c in s]
    assert len(chars) == 3


main()
