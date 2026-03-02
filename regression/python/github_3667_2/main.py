def test_nested_lists_and_mutability():
    nested = [[1], [2]]
    shallow = nested.copy()

    nested[0].append(99)
    assert shallow[0] == [1, 99]  # shallow copy shares inner list

test_nested_lists_and_mutability()
