# Stubs for random module
class random:
    @classmethod
    def randint(a:int, b:int) -> int:
        value:int = nondet_int()
        __ESBMC_assume(value >= a and value <= b)
        return value  # Ensures value is within [a, b]

    @classmethod
    def random() -> float:
        value:float = nondet_float()
        __ESBMC_assume(value >= 0.0 and value < 1.0)
        return value #  Returns a floating number [0,1.0).