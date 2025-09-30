import random

def esbmc_choice(seq: list[int]) -> int:
    assert len(seq) > 0
    
    # Get a nondeterministic index
    idx:int = random.randint(1,4)
    
    # Restrict the nondet index to valid range
    __ESBMC_assume(idx > 0 and idx < len(seq))
    
    return seq[idx]

# Example usage:
seq = [10, 20, 30, 40]
choice = esbmc_choice(seq)
assert choice == 10 or choice == 20 or choice == 30 or choice == 40
