// __ESBMC_old(r[j]) on a pointer parameter, quantified DIRECTLY IN ENSURES,
// under --replace-call-with-contract.
//
// Half of this is now fixed: materialize_old_snapshots_at_callsite no
// longer hard-rejects a region snapshot outright. find_callsite_is_fresh_
// extent locates the callee's own __ESBMC_is_fresh(r, N) clause and rebinds
// N to the caller's actual argument, giving a call-site-visible extent, and
// the existing enforce-mode copy-loop builder (materialize_ptr_region_old_
// snapshot, generalised to take that extent directly) fills a real
// array-typed temp with it, exactly as it already does for the same
// quantified reference when it appears only in a loop_invariant instead
// (see github_4219_old_in_forall_array_param_replace_mode_loopinv_pass,
// which exercises exactly that shape and is fully fixed).
//
// This exact test is not: when the SAME quantified reference is used
// directly inside ensures's own forall (rather than only in a
// loop_invariant, which replace-mode never emits any check from at all),
// the built region-snapshot array, once referenced from inside the
// quantifier body of an ASSUME at the call site, hits a separate,
// deeper issue during SMT encoding -- "Non-pointer op being interpreted as
// pointer without cast" -- rather than the GOTO-conversion-time abort this
// test originally documented. Root cause not yet isolated; still open
// (#7057).

#define N 4

void bump(int r[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, N * sizeof(int)));
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < N) || (r[j] == __ESBMC_old(r[j]) + 1)));
  __ESBMC_assigns(r);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + 1;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4};
  bump(arr);
  return 0;
}
