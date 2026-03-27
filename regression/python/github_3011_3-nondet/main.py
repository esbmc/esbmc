def foo(arr: list[int]) -> None:
    if len(arr) <= 1:
        for i in arr:
            pass


# Valid calls
foo([])        # empty list
foo([1])       # single element
foo([1, 2])    # more than one element (if-block won't execute)
