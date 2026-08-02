def main() -> None:
    # float.is_integer() on a constant literal receiver. A variable receiver
    # already routes through the float operational model; a bare literal did
    # not, reporting a spurious "Unsupported function" before this fix.
    assert (2.0).is_integer() is True
    assert (2.5).is_integer() is False
    assert (0.0).is_integer() is True

    # Unary +/- literals work; sign does not affect integrality.
    assert (-4.0).is_integer() is True
    assert (-4.5).is_integer() is False
    assert (+8.0).is_integer() is True

    # The bool result composes in boolean context.
    if (10.0).is_integer():
        ok = 1
    else:
        ok = 0
    assert ok == 1

    # The variable receiver still works (unchanged path).
    x = 3.0
    assert x.is_integer() is True


main()
