# isinstance() on an element of a list parameter mixing a list and an int
# follows the element read, not the first element's type (#8472).
def first_is_list(a):
    return isinstance(a[0], list)


def second_is_int(a):
    x = a[1]
    return isinstance(x, int) and not isinstance(x, list)


assert first_is_list([[1], 2])
assert second_is_int([[1], 2])
