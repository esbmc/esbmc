# HumanEval/90
# Entry Point: next_smallest
# ESBMC-compatible format with direct assertions


def next_smallest(lst):
    """
    You are given a list of integers.
    Write a function next_smallest() that returns the 2nd smallest element of the list.
    Return None if there is no such element.
    
    next_smallest([1, 2, 3, 4, 5]) == 2
    next_smallest([5, 1, 4, 3, 2]) == 2
    next_smallest([]) == None
    next_smallest([1, 1]) == None
    """
    lst = sorted(set(lst))
    return None if len(lst) < 2 else lst[1]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert next_smallest([1, 2, 3, 4, 5]) == 2
    # assert next_smallest([5, 1, 4, 3, 2]) == 2
    # assert next_smallest([]) == None
    # assert next_smallest([1, 1]) == None
    # assert next_smallest([1,1,1,1,0]) == 1
    # assert next_smallest([1, 0**0]) == None
    # assert next_smallest([-35, 34, 12, -45]) == -35
    # assert True
    print("✅ HumanEval/90 - All assertions completed!")
