def test() -> None:
    arr: list[int] = [1, 2, 3]
    copied: list[int] = arr.copy()

    # This should fail - copy is independent
    copied[0] = 99
    assert arr[0] == 99


test()
