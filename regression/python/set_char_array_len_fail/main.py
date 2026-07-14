# Negative variant: the char-array set has 3 unique elements, so asserting 4
# must FAIL — pins the array-length bound built in IREP2 (python_set.cpp).
def main() -> None:
    s = "abc"
    t = set(s)
    assert len(t) == 4


main()
