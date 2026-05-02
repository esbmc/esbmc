# Stubs for random module.
# See https://docs.python.org/3/library/random.html


def randint(a: int, b: int) -> int:
    value: int = nondet_int()
    __ESBMC_assume(value >= a and value <= b)
    return value  # Ensures value is within [a, b]


def random() -> float:
    value: float = nondet_float()
    __ESBMC_assume(value >= 0.0 and value < 1.0)
    return value  #  Returns a floating number [0,1.0).


def uniform(a: float, b: float) -> float:
    value: float = nondet_float()
    if a <= b:
        __ESBMC_assume(value >= a and value <= b)
    else:
        __ESBMC_assume(value >= b and value <= a)
    return value


def getrandbits(k: int) -> int:
    if k < 0:
        raise ValueError("number of bits must be non-negative")

    if k == 0:
        return 0

    value: int = nondet_int()
    max_val: int = (1 << k) - 1  # 2**k - 1
    __ESBMC_assume(value >= 0 and value <= max_val)
    return value


def randrange(start: int, stop: int = None, step: int = 1) -> int:
    # Handle the single-argument case: randrange(stop)
    if stop is None:
        actual_start: int = 0
        actual_stop: int = start
        actual_step: int = 1
    else:
        actual_start: int = start
        actual_stop: int = stop
        actual_step: int = step

    # Validate step
    if actual_step == 0:
        raise ValueError("step argument must not be zero")

    # Calculate count based on step direction
    if actual_step > 0:
        count: int = (actual_stop - actual_start + actual_step - 1) // actual_step
    else:
        count: int = (actual_stop - actual_start + actual_step + 1) // actual_step

    # Validate that range is non-empty
    if count <= 0:
        raise ValueError("empty range for randrange()")

    # Select random index
    index: int = nondet_int()
    __ESBMC_assume(index >= 0 and index < count)

    return actual_start + (index * actual_step)
