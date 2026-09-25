// Structural counterpart to github_4219_old_in_forall_array_param_replace_
// mode_loopinv_{pass,fail} (identical contract and callee), added because
// neither of those tests' VERDICTS actually depends on the region
// snapshot's extent being correct: under this contract's exact shape, the
// copy loop materialize_old_snapshots_at_callsite builds is read only from
// loop_invariant, which --replace-call-with-contract never checks, so the
// snapshot is sliced away before SMT encoding regardless of whether the
// extent computed for it is right or wrong. Confirmed directly by mutation:
// forcing find_callsite_is_fresh_extent's resolved extent to a constant
// 1 byte (instead of the correct, rebound N * sizeof(int)) leaves the
// entire function_contract suite passing, _loopinv_pass and _loopinv_fail
// included.
//
// This test instead inspects the emitted GOTO program directly, via
// --goto-functions-only, BEFORE slicing removes the snapshot -- the
// region-copy loop's own exit condition names its element count literally
// (`old_region_i_bump_call_<N> < 5`), which the required pattern below
// pins. The same 1-byte-extent mutation that leaves every end-to-end
// verdict unchanged flips this bound to `< 1`, making this the one test in
// the suite that actually fails if the resolved extent is wrong rather
// than merely absent.
#define N 5

void bump(int r[N], int c)
{
  unsigned k;
  __ESBMC_requires(__ESBMC_is_fresh(r, N * sizeof(int)));
  __ESBMC_ensures(r[0] == __ESBMC_old(r[0]) + c);
  __ESBMC_ensures(r[1] == __ESBMC_old(r[1]) + c);
  __ESBMC_ensures(r[2] == __ESBMC_old(r[2]) + c);
  __ESBMC_ensures(r[3] == __ESBMC_old(r[3]) + c);
  __ESBMC_ensures(r[4] == __ESBMC_old(r[4]) + c);
  __ESBMC_assigns(r[0], r[1], r[2], r[3], r[4]);

  unsigned i;
  __ESBMC_loop_invariant(
    i <= N &&
    __ESBMC_forall(&k, !(k < i) || (r[k] == __ESBMC_old(r)[k] + c)) &&
    __ESBMC_forall(&k, !(k >= i && k < N) || (r[k] == __ESBMC_old(r)[k])));
  for (i = 0; i < N; i++)
    r[i] = r[i] + c;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4, 5};
  bump(arr, 2);
  __ESBMC_assert(
    arr[0] == 3 && arr[1] == 4 && arr[2] == 5 && arr[3] == 6 && arr[4] == 7,
    "ensures propagated the correct incremented values to the caller");
  return 0;
}
