def main() -> None:
    # Zero-arg int methods on a constant literal receiver. A variable receiver
    # already routes through the int operational model; a bare literal did not,
    # reporting a spurious "Unsupported function" before this fix.
    assert (255).bit_length() == 8
    assert (0).bit_length() == 0
    assert (1024).bit_length() == 11
    assert (7).bit_count() == 3
    assert (5).conjugate() == 5

    # CPython takes the magnitude for bit_length/bit_count (sign ignored), so a
    # unary-minus literal works too; conjugate of an int is the int itself.
    assert (-5).bit_length() == 3
    assert (-7).bit_count() == 3
    assert (-5).conjugate() == -5
    assert (+8).bit_length() == 4

    # The result is an int and composes in arithmetic.
    assert (255).bit_length() + 1 == 9

    # The variable receiver still works (unchanged path).
    x = 1024
    assert x.bit_length() == 11


main()
