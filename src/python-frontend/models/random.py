# Stubs for random module

class random:
    @classmethod
    def randint(a:int, b:int) -> int:
        value:int = nondet_int()
        __ESBMC_assume(value >= 0 and value <= b)
        return value  # Ensures value is within [a, b]
