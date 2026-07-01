# Negative variant: the comprehension yields 3 chars, so asserting 4 must FAIL
# — pins the IREP2 loop-condition bound (list_comprehension.cpp).
def main() -> None:
    s = "abc"
    chars = [c for c in s]
    assert len(chars) == 4


main()
