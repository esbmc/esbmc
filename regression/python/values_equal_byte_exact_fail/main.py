# The negative direction: two values differing only in the most significant
# byte must not compare equal.
def main():
    high = 1 << 56
    ys = [0]
    assert high in ys


main()
