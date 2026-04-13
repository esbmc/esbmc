def test_list_properties():
    # Test 1: List equality
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    assert list1 == list2, f"Expected {list1} to equal {list2}"

    # Test 2: List length
    assert len(list1) == 3, f"Expected length of {list1} to be 3"

    # Test 3: List contains an element
    assert 2 in list1, f"Expected 2 to be in {list1}"

    # Test 4: List does not contain an element
    assert 4 not in list1, f"Expected 4 not to be in {list1}"

    # Test 5: List element type
    # assert all(isinstance(x, int) for x in list1), f"Expected all elements in {list1} to be integers"

    # Test 6: List sorting
    unsorted_list = [3, 1, 2]
    sorted_list = sorted(unsorted_list)
    assert sorted_list == [1, 2, 3], f"Expected sorted version of {unsorted_list} to be [1, 2, 3]"

    # Test 7: List concatenation
    list3 = [4, 5]
    combined_list = list1 + list3
    assert combined_list == [1, 2, 3, 4, 5], f"Expected {list1} + {list3} to be [1, 2, 3, 4, 5]"

    # Test 8: List replication
    replicated_list = list3 * 2
    assert replicated_list == [4, 5, 4, 5], f"Expected {list3} * 2 to be [4, 5, 4, 5]"

    # Test 9: List slicing
    sublist = list1[1:3]
    assert sublist == [2, 3], f"Expected slice of {list1} from index 1 to 3 to be [2, 3]"

    # Test 10: Empty list
    empty_list = []
    assert len(empty_list) == 0, "Expected empty list to have length 0"

    print("All list property tests passed.")

test_list_properties()
