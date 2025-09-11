import random


def secure_choice(seq: list[int]) -> int:
    assert (len(seq) > 0)
    idx = nondet_int()
    __ESBMC_assume(idx >= 0 and idx < len(seq))
    return seq[idx]


# Example usage:
seq = [10, 20, 30, 40]
choice = secure_choice(seq)
assert (choice == 40)
