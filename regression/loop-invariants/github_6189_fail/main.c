/*
 * Negative variant of issue #6189: same shape, but the second invariant
 * asserts the wrong value for the function call. With hi_idx <= SIZE assumed
 * first, loop(hi_idx) is now properly bounded and evaluates to hi_idx, so the
 * inductive step ASSERT of loop(hi_idx) == hi_idx + 1 must fail. This guards
 * against the reordering vacuously accepting a wrong invariant.
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
  __ESBMC_loop_invariant(loop(hi_idx) == hi_idx + 1);
  while (hi_idx < SIZE)
    ++hi_idx;
  __ESBMC_assert(hi_idx == SIZE, "end: hi_idx");
}
