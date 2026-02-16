def test_sum_of_positive_elements():
    """Verify sum of positive elements is positive."""
    x = nondet_list(4)
    
    # Assume all elements are positive and list is non-empty
    if len(x) > 0:
        all_positive = True
        for elem in x:
            if elem <= 0:
                all_positive = False
                break
        
        if all_positive:
            total = 0
            for elem in x:
                total += elem
            assert total > 0

test_sum_of_positive_elements()
