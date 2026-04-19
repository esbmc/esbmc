def test_max_element_property():
    """Verify max element is >= all elements."""
    x:list[int] = nondet_list(4)
    
    if len(x) > 0:
        max_val = x[0]
        for i in range(1, len(x)):
            if x[i] > max_val:
                max_val = x[i]
        
        # Verify max_val is indeed the maximum
        for elem in x:
            assert max_val >= elem

test_max_element_property()
