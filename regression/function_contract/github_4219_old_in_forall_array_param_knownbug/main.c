// An `int r[N]` parameter is a pointer, so `bound_array_element` finds no
// array-typed base and the lift declines before it starts. The quantified
// __ESBMC_old then reaches the frontend unchanged and reports "cannot model
// call to __ESBMC_old_raw on a quantified variable".
//
// The lift is right to decline. A quantified index cannot be snapshotted one
// element at a time -- the snapshot is taken once, before the body runs, while
// j still ranges -- so the whole region has to be copied and indexed after. For
// a pointer parameter that region is reached through the pointer and its extent
// lives in the `is_fresh` clause, which the lift never sees.
//
// Wrapping the array in a struct (github_4219_old_in_forall) or naming it
// directly (github_4219_old_in_forall_array_base) both work, because both give
// the snapshot a named object whose address resolves.
#define N 4
#define BOUND 100

void bump(int r[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, N * sizeof(int)));
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < N) || (r[j] > -BOUND && r[j] < BOUND)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < N) || (r[j] == __ESBMC_old(r[j]) + 1)));
  __ESBMC_assigns(r);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + 1;
}

int main(void)
{
  return 0;
}
