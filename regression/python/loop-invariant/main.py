def main() -> None:
    x: int = 0
    y: int = 50
    # The loop keeps x between 0 and 100
    __ESBMC_assume(0 <= x and x <= 100)
    # If x is 50 or less then y is 50
    __ESBMC_assume(not (x <= 50) or y == 50)
    # If x is greater than 50 then y = x
    __ESBMC_assume(not (x >  50) or y == x)
    # At the end of the loop x is not <100
    __ESBMC_assume(x >= 100)
    assert(y == 100)

main()
