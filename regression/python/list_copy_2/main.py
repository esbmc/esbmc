def test() -> None:
    original: list[int] = [1, 2, 3]
    copied: list[int] = original.copy()

    # Modify the copy
    copied[0] = 99

    # Original should be unchanged
    assert original[0] == 1
    assert copied[0] == 99


test()
