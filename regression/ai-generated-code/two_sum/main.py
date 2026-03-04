"""
Two Sum — verified with ESBMC.

Given an array of integers nums and an integer target, return the indices
of two distinct elements whose sum equals target.  Returns [] when no
solution exists.

Run with:
  esbmc two_sum.py --unwindset 107:5,108:5,109:5,110:5 --no-pointer-check --k-induction
"""

from typing import List
from esbmc import nondet_int, __ESBMC_assume


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def two_sum(nums: List[int], n: int, target: int) -> List[int]:
    """
    Brute-force O(n²) two-sum.

    Preconditions (caller's responsibility):
      - n >= 2
      - n == len(nums)

    Postconditions (verified below):
      - Returns [] OR [i, j] where:
          0 <= i < j < n
          nums[i] + nums[j] == target
    """
    i: int = 0
    while i < n:
        # --- Outer loop invariant ---
        # i stays in [0, n): loop variable is non-negative and bounded
        assert i >= 0 and i < n, "INV outer: i in [0, n)"

        j: int = i + 1
        while j < n:
            # --- Inner loop invariant ---
            # j stays in (i, n): strictly greater than i, within bounds
            assert j > i and j < n, "INV inner: i < j < n"

            if nums[i] + nums[j] == target:
                return [i, j]
            j = j + 1
        i = i + 1
    return []


# ---------------------------------------------------------------------------
# ESBMC verification harness
# ---------------------------------------------------------------------------

def test_two_sum() -> None:
    """Verify two_sum against all symbolic inputs of size up to 4."""

    # --- symbolic array length ---
    n: int = nondet_int()
    __ESBMC_assume(n >= 2 and n <= 4)

    # --- symbolic array elements (flat, bounded) ---
    a0: int = nondet_int()
    a1: int = nondet_int()
    a2: int = nondet_int()
    a3: int = nondet_int()
    __ESBMC_assume(a0 >= -10 and a0 <= 10)
    __ESBMC_assume(a1 >= -10 and a1 <= 10)
    __ESBMC_assume(a2 >= -10 and a2 <= 10)
    __ESBMC_assume(a3 >= -10 and a3 <= 10)

    nums: List[int] = [a0, a1, a2, a3]

    # --- symbolic target ---
    target: int = nondet_int()
    __ESBMC_assume(target >= -20 and target <= 20)

    # --- call the function ---
    result: List[int] = two_sum(nums, n, target)

    # Property 0: result length is always 0 or 2
    assert len(result) == 0 or len(result) == 2, "Result length is 0 or 2"

    if len(result) == 2:
        idx_i: int = result[0]
        idx_j: int = result[1]

        # Property 1: both indices are within bounds
        assert idx_i >= 0 and idx_i < n, "First index in bounds"
        assert idx_j >= 0 and idx_j < n, "Second index in bounds"

        # Property 2: indices are distinct
        assert idx_i != idx_j, "Indices are distinct"

        # Property 3: the pair sums to target
        assert nums[idx_i] + nums[idx_j] == target, "Sum equals target"

        # Property 4: first index is always strictly less (canonical order)
        assert idx_i < idx_j, "Indices returned in ascending order"

    # Property 5: if a valid pair exists in the array, result must not be empty
    found: bool = False
    p: int = 0
    while p < n:
        # --- Completeness outer invariant ---
        assert p >= 0 and p < n, "INV completeness outer: p in [0, n)"

        q: int = p + 1
        while q < n:
            # --- Completeness inner invariant ---
            assert q > p and q < n, "INV completeness inner: p < q < n"

            if nums[p] + nums[q] == target:
                found = True
            q = q + 1
        p = p + 1

    if found:
        assert len(result) == 2, "Solution exists but was not returned"


test_two_sum()
