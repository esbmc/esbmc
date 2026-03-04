"""
Is Sorted (Non-Decreasing) — verified with ESBMC.

Given an integer array nums of length n, return True if the array is sorted
in non-decreasing order, i.e. nums[i] <= nums[i+1] for all i in [0, n-2],
and False otherwise.

Run with:
  esbmc is_sorted.py --unwind 5 --no-pointer-check --bitwuzla
"""

from typing import List
from esbmc import nondet_int, __ESBMC_assume


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def is_sorted(nums: List[int], n: int) -> bool:
    """
    Return True iff nums[0..n-1] is sorted in non-decreasing order.

    Preconditions (caller's responsibility):
      - n >= 1
      - n == len(nums)

    Postconditions (verified below):
      - Returns True  => nums[i] <= nums[i+1] for all i in [0, n-2]
      - Returns False => exists i in [0, n-2] such that nums[i] > nums[i+1]
    """
    i: int = 0
    while i < n - 1:
        # Loop invariant 1: i stays in [0, n-1)
        assert i >= 0 and i < n - 1, "INV: i in [0, n-1)"

        if nums[i] > nums[i + 1]:
            return False

        # Loop invariant 2: all adjacent pairs seen so far are non-decreasing
        assert nums[i] <= nums[i + 1], "INV: nums[i] <= nums[i+1] when continuing"

        i = i + 1

    return True


# ---------------------------------------------------------------------------
# ESBMC verification harness
# ---------------------------------------------------------------------------

def test_is_sorted() -> None:
    """Verify is_sorted for all symbolic arrays of size up to 4."""

    # --- symbolic array length ---
    n: int = nondet_int()
    __ESBMC_assume(n >= 1 and n <= 4)

    # --- symbolic array elements (bounded) ---
    a0: int = nondet_int()
    a1: int = nondet_int()
    a2: int = nondet_int()
    a3: int = nondet_int()
    __ESBMC_assume(a0 >= -10 and a0 <= 10)
    __ESBMC_assume(a1 >= -10 and a1 <= 10)
    __ESBMC_assume(a2 >= -10 and a2 <= 10)
    __ESBMC_assume(a3 >= -10 and a3 <= 10)

    nums: List[int] = [a0, a1, a2, a3]

    # --- call the function under test ---
    result: bool = is_sorted(nums, n)

    # --- compute the reference result independently (no loops) ---
    # A single-element array is always sorted.
    expected: bool = True
    if n >= 2 and a0 > a1:
        expected = False
    if n >= 3 and a1 > a2:
        expected = False
    if n >= 4 and a2 > a3:
        expected = False

    # Property 1 — Correctness: result always matches the reference
    assert result == expected, "Result matches expected sorted flag"

    # Property 2 — Soundness: True => every adjacent pair is non-decreasing
    if result:
        if n >= 2:
            assert a0 <= a1, "Sorted => nums[0] <= nums[1]"
        if n >= 3:
            assert a1 <= a2, "Sorted => nums[1] <= nums[2]"
        if n >= 4:
            assert a2 <= a3, "Sorted => nums[2] <= nums[3]"

    # Property 3 — Completeness: False => at least one inversion exists
    if not result:
        inversion_exists: bool = False
        if n >= 2 and a0 > a1:
            inversion_exists = True
        if n >= 3 and a1 > a2:
            inversion_exists = True
        if n >= 4 and a2 > a3:
            inversion_exists = True
        assert inversion_exists, "Not sorted => inversion exists"


test_is_sorted()
