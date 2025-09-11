def comprehensive_list_test():
    # Create two separate lists with identical content
    x = [1, 2, 3]
    y = [1, 2, 3]

    # Test 1: Different objects
    assert x is not y, "x and y should be different objects"

    # Test 2: Reference assignment
    z = y

    assert y is z, "y and z should reference the same object"
    assert z == y, "z and y should have equal content"

    # Test 3: Transitivity check
    assert x is not z, "x and z should be different objects"

    # Test 4: Modification affects references
    w = [1, 2, 3, 4]
    a = w
    assert w is a, "w and z should still reference the same object"
    assert w == a, "y and z should still have equal content"
    assert x != w, "x should now have different content from y"


comprehensive_list_test()
