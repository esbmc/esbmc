import math

def test_math_functions() -> None:
    # Test floor and ceil functions
    assert math.floor(3.7) == 3
    assert math.ceil(3.2) == 4
    
    # Test combinatorial function
    assert math.comb(5, 2) == 10  # C(5,2) = 5!/(2!*3!) = 10
    
    # Test symmetry property of combinations
    n = 6
    k = 2
    assert math.comb(n, k) == math.comb(n, n - k)
    
    # Test special value detection
    nan_value = float('nan')
    inf_value = float('inf')
    
    assert math.isnan(nan_value) == True
    assert math.isinf(inf_value) == True
    assert math.isnan(5.0) == False

test_math_functions()
