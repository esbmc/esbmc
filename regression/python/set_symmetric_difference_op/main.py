def main() -> None:
    # The ^ operator on sets is symmetric difference: (a - b) | (b - a). It was
    # unmapped (only - / & / | were), so it fell through to a bitwise path and
    # produced the wrong result.
    assert {1, 2, 3} ^ {2, 3, 4} == {1, 4}

    # Augmented assignment s ^= other lowers to the same operator.
    s = {1, 2, 3}
    s ^= {2, 3, 4}
    assert s == {1, 4}

    # The symmetric_difference() method (variable receiver) is unchanged.
    a = {1, 2, 3}
    assert a.symmetric_difference({2, 3, 4}) == {1, 4}


main()
