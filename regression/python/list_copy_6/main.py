def test() -> None:
    original: list[int] = [1, 2, 3]
    copied: list[int] = original.copy()

    # Append to copy
    copied.append(4)

    # Original should have 3 elements, copy should have 4
    assert len(original) == 3
    assert len(copied) == 4
    assert copied[3] == 4


test()
