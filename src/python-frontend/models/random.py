# pylint: disable=undefined-variable,chained-comparison,unused-argument
# `nondet_int`, `nondet_float`, and `__ESBMC_assume` are ESBMC
# intrinsics matched by name by the Python converter; they have no
# Python binding. Chained comparisons are intentionally written in
# the `a >= x and a <= y` form because they are arguments to
# `__ESBMC_assume` and we keep them in the same shape as the
# corresponding C operational models. seed(a) and shuffle(lst) keep
# their CPython parameter names for API compatibility but discard the
# value: nondet outputs already cover all seeds, and shuffle is
# modelled as an under-approximation that leaves the list untouched.

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


def seed(a: int = 0) -> None:
    """random.seed(a) — no-op stub.

    The non-determinism of nondet_int / nondet_float already covers any
    seed-dependent outcome the program could observe, so seeding has no
    observable effect under verification.
    """
    return


def choice(seq: list[int]) -> int:
    """random.choice(seq) — return a non-deterministic element of `seq`.

    Models the choice as `seq[i]` for a constrained nondet index. Raises
    IndexError on an empty sequence, matching CPython.
    """
    n: int = len(seq)
    if n <= 0:
        raise IndexError("Cannot choose from an empty sequence")
    i: int = nondet_int()
    __ESBMC_assume(i >= 0 and i < n)
    return seq[i]


def shuffle(lst: list[int]) -> None:
    """random.shuffle(lst) — permutes `lst` in place.

    Under-approximation: leaves the list untouched. A precise model would
    nondet-shuffle the elements, but doing so requires expressing
    permutation constraints over `lst`, which is out of scope here.
    """
    return


def sample(population: list[int], k: int) -> list[int]:
    """random.sample(population, k) — k distinct elements from population.

    Returns the first `k` elements of `population` as an under-approximation;
    a precise model would pick `k` distinct nondet indices.
    """
    n: int = len(population)
    if k < 0 or k > n:
        raise ValueError("Sample larger than population or is negative")
    result: list[int] = []
    i: int = 0
    while i < k:
        result.append(population[i])
        i = i + 1
    return result


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
