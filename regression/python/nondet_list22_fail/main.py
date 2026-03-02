def test_list_elements():
    lst = nondet_list(3)  # Expected: a list with up to 3 elements

    if len(lst) >= 2:
        # Assertion: the first two elements of the list should be able to differ
        assert lst[0] == lst[1]  # Expect this assertion to be violated
test_list_elements()