def foo(arr: list[int]) -> None:
    if len(arr) <= 1:
        if len(arr) == 1:
            assert arr[0] == arr[0]  # safe access
        for i in arr:
            assert i == arr[0]
