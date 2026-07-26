/*
 * Regression test for issue #6189: a loop invariant that calls a function
 * whose own loop is bounded by a havoc'd variable, where that variable is
 * constrained by a *separate* invariant.
 *
 * In the inductive step hi_idx is havoc'd, so loop(hi_idx) can only be
 * bounded once hi_idx <= SIZE is assumed. The instrumentation must assume the
 * pure constraint hi_idx <= SIZE before evaluating loop(hi_idx); otherwise the
 * inner loop is symex'd with an unconstrained bound and its unwinding
 * assertion fails spuriously. Ordering of the two invariants must not matter.
 */
#define SIZE 5
typedef int idx_t;

int loop(idx_t hi_idx)
{
  int res = 0;
  idx_t idx;
#pragma unroll SIZE + 1
  for (idx = 0; idx < hi_idx; ++idx)
    ++res;
  return res;
}

int main()
{
  idx_t hi_idx = 0;
  __ESBMC_loop_invariant(hi_idx <= SIZE);
  __ESBMC_loop_invariant(loop(hi_idx) == hi_idx);
  while (hi_idx < SIZE)
    ++hi_idx;
  __ESBMC_assert(hi_idx == SIZE, "end: hi_idx");
}
