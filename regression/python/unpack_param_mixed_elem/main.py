# Call sites that disagree on the element type each get a copy of the
# function typed for theirs, so the unpacking reads typed elements (#9545).


def head_rest(arr):
    if arr:
        first, *rest = arr
        return len(rest)
    return 0


assert head_rest([1, 2]) == 1
assert head_rest(["a", "b"]) == 1
