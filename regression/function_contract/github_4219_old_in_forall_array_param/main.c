// An `int r[N]` parameter is a pointer, so `bound_array_element` used to
// find no array-typed base and decline before starting -- the quantified
// __ESBMC_old reached the frontend unchanged and reported "cannot model
// call to __ESBMC_old_raw on a quantified variable" (#7057).
//
// A quantified index cannot be snapshotted one element at a time -- the
// snapshot is taken once, before the body runs, while j still ranges -- so
// the whole region has to be copied and indexed after. For a pointer
// parameter that region is reached through the pointer, with its extent
// coming from the `is_fresh` clause: `bound_array_element` now also
// recognises a pointer-parameter base, and once contracts.cpp resolves the
// extent, materialize_old_snapshots_at_wrapper fills a real array-typed
// temp with an explicit copy loop instead of the single whole-value ASSIGN
// the named-array case (github_4219_old_in_forall_array_base) uses.
//
// See also github_4219_old_in_forall_array_param_fail (same contract, a
// broken implementation) and github_4219_old_in_forall_ptr_to_array_knownbug
// (the related but distinct `int (*r)[N]` shape, still open).
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
