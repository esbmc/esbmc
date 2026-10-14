# Negative variant: the inline optional comparison must still be able to fail
# (no spurious SUCCESSFUL). second_or_none([1,2,3]) is 2, not 99.


def second_or_none(lst):
    return None if len(lst) < 2 else lst[1]


assert second_or_none([1, 2, 3]) == 99
