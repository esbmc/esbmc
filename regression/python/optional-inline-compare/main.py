# Comparing the result of an inline call that returns an optional (None | T)
# directly against a value must not abort the frontend. The call result is a
# code_function_callt; taking its ".value" member without materialising it
# first aborted goto generation (member2t requires a struct source). See the
# member2t crash behind humaneval/90 (#4807).


def second_or_none(lst):
    return None if len(lst) < 2 else lst[1]


# Inline form (the crashing one): the call is not bound to a variable first.
assert second_or_none([1, 2, 3]) == 2
assert second_or_none([10, 20]) == 20

# Bound form (already worked) must still agree.
r = second_or_none([7, 8, 9])
assert r == 8
