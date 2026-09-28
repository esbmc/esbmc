# Unpacking a parameter shows it is a sequence, but an unannotated parameter
# stayed Any, so the unpacking was handed a bare pointer and refused. The
# element type the call sites agree on is enough to type it -- an empty-list
# call contributes nothing and does not block agreement.


def head_rest(arr):
    if arr:
        first, *rest = arr
        return len(rest)
    return 0


assert head_rest([1, 2]) == 1
assert head_rest([1, 2, 3]) == 2
assert head_rest([]) == 0
