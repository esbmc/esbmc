"""
Maximum Prefix Sum — verified with ESBMC.

Given an integer array nums of length n, return the largest sum among all
contiguous prefixes starting at index 0:

    max{ sum(nums[0..k]) | k in [0, n-1] }

Run with:
  esbmc max_prefix_sum.py --unwind 5 --no-pointer-check --bitwuzla
"""

from typing import List
from esbmc import nondet_int, __ESBMC_assume


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def max_prefix_sum(nums: List[int], n: int) -> int:
    """
    Compute the maximum prefix sum over nums[0..n-1].

    Preconditions (caller's responsibility):
      - n >= 1
      - n == len(nums)

    Postcondition (verified below):
      - Returns max{ sum(nums[0..k]) | k in [0, n-1] }

    Algorithm: scan left-to-right accumulating the running prefix sum;
    keep track of the largest value seen.
    """
    # Initialise with the only prefix that always exists: prefix[0] = nums[0]
    current_sum: int = nums[0]
    max_sum: int = nums[0]
    i: int = 1
    while i < n:
        # Loop invariant 1: i stays in [1, n)
        assert i >= 1 and i < n, "INV: i in [1, n)"

        current_sum = current_sum + nums[i]
        if current_sum > max_sum:
            max_sum = current_sum

        # Loop invariant 2: max_sum is always >= the running prefix sum
        assert max_sum >= current_sum, "INV: max_sum >= current prefix sum"

        i = i + 1

    return max_sum


# ---------------------------------------------------------------------------
# ESBMC verification harness
# ---------------------------------------------------------------------------

def test_max_prefix_sum() -> None:
    """Verify max_prefix_sum against all symbolic inputs of size up to 3."""

    # --- symbolic array length ---
    n: int = nondet_int()
    __ESBMC_assume(n >= 1 and n <= 3)

    # --- symbolic array elements (bounded) ---
    a0: int = nondet_int()
    a1: int = nondet_int()
    a2: int = nondet_int()
    __ESBMC_assume(a0 >= -10 and a0 <= 10)
    __ESBMC_assume(a1 >= -10 and a1 <= 10)
    __ESBMC_assume(a2 >= -10 and a2 <= 10)

    nums: List[int] = [a0, a1, a2]

    # --- call the function under test ---
    result: int = max_prefix_sum(nums, n)

    # --- compute the reference result independently ---
    # Build each prefix sum step by step (no loops → directly encodable by BMC)
    expected: int = a0          # prefix[0] always exists
    prefix: int = a0

    if n >= 2:
        prefix = prefix + a1    # prefix[1]
        if prefix > expected:
            expected = prefix

    if n >= 3:
        prefix = prefix + a2    # prefix[2]
        if prefix > expected:
            expected = prefix

    # Property 1 — Correctness: result equals the true maximum prefix sum
    assert result == expected, "Result equals maximum prefix sum"

    # Property 2 — Upper bound: result does not exceed any known prefix sum
    assert result >= a0, "Result >= prefix[0]"
    if n >= 2:
        assert result >= a0 + a1, "Result >= prefix[1]"
    if n >= 3:
        assert result >= a0 + a1 + a2, "Result >= prefix[2]"


test_max_prefix_sum()
