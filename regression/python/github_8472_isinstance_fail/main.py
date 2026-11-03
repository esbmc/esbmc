def first_is_list(a):
    return isinstance(a[0], list)


assert not first_is_list([[1], 2])
