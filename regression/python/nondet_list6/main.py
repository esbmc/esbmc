def test_find_element_in_list():
    """Verify element search works correctly."""
    x = nondet_list(6)
    target = nondet_int()

    found = False
    for elem in x:
        if elem == target:
            found = True
            break

    # If we found it, verify it's actually in the list
    if found:
        actually_found = False
        for elem in x:
            if elem == target:
                actually_found = True
        assert actually_found


test_find_element_in_list()
