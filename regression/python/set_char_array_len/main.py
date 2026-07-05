# set() over a char-array variable (not a string literal, which takes the
# fast path) drives the array-length branch in python_set.cpp whose `size - 1`
# bound is built in IREP2 with gen_typecast_arithmetic width reconciliation.
def main() -> None:
    s = "abc"
    t = set(s)
    assert len(t) == 3


main()
