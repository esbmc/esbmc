# Call sites that disagree on the element type give no usable type for the
# unpacked parameter. The unpacking refuses it rather than reading the
# elements untyped and reporting a verdict that does not hold.


def head_rest(arr):
    if arr:
        first, *rest = arr
        return len(rest)
    return 0


assert head_rest([1, 2]) == 1
assert head_rest(["a", "b"]) == 1
