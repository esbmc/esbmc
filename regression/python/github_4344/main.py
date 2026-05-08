def main() -> None:
    # Tuple variable.
    t = (False, True, False)
    assert any(t)
    assert not all(t)

    # Tuple literal in argument position.
    assert any((False, True))
    assert all((True, True))

    # Set literal.
    assert any({False, True})

    # Set variable.
    s = {False, True, False}
    assert any(s)

    # Empty cases.
    assert not any(())
    assert all(())
    assert not any(set())
main()
