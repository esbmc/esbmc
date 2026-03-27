def analyze_paths(x: int) -> str:
    """Function with multiple execution paths"""
    if x > 100:
        return "high"
    elif x > 50:
        return "medium"
    elif x > 0:
        return "low"
    else:
        return "zero_or_negative"

def test_reachability() -> None:
    x: int = nondet_int()
    
    # Cover: Can x be greater than 100? (Expected: FAIL - satisfiable)
    __ESBMC_cover(x > 100)
    
    # Cover: Can x be in the medium range? (Expected: FAIL - satisfiable)
    __ESBMC_cover(50 < x <= 100)
    
    # Add constraint: x must be positive
    __ESBMC_assume(x > 0)
    
    # Cover: Can x still be negative? (Expected: SUCCESS - not satisfiable)
    __ESBMC_cover(x < 0)
    
    # Cover: Can x be positive? (Expected: FAIL - satisfiable)
    __ESBMC_cover(x > 0)
    
    result = analyze_paths(x)
    
    # Cover: Can we reach the "high" path? (Expected: FAIL - satisfiable)
    __ESBMC_cover(result == "high")
    
    # Cover: Can we reach the "zero_or_negative" path? (Expected: SUCCESS - not satisfiable after assume)
    __ESBMC_cover(result == "zero_or_negative")

def test_dead_code_detection() -> None:
    """Using cover to identify unreachable code"""
    value: int = nondet_int()
    __ESBMC_assume(value >= 0)
    
    if value < 0:
        # This is dead code - the cover should succeed (prove unreachability)
        __ESBMC_cover(True)
        print("This branch is unreachable")
    
    if value >= 0:
        # This is reachable code - the cover should fail (show reachability)
        __ESBMC_cover(True)
        print("This branch is reachable")

test_reachability()
test_dead_code_detection()

