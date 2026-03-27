def test_cover_dead_code():
    x: int = nondet_int()
    __ESBMC_assume(x > 0)
    
    if x < 0:
        # This branch is dead code
        # Cover will fail here (not satisfiable)
        __ESBMC_cover(True)
        print("This is unreachable")
    
    if x > 0:
        # This branch is reachable
        # Cover will succeed here (satisfiable)
        __ESBMC_cover(True)
        print("This is reachable")

test_cover_dead_code()
