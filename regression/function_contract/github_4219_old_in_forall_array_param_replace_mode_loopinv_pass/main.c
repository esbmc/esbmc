// __ESBMC_old(r[j]) on a pointer parameter, quantified inside a
// loop_invariant (needed for the function's OWN function-level inductive
// proof, not by ensures itself), under --replace-call-with-contract.
//
// Before this fix, GOTO conversion aborted unconditionally the moment ANY
// region snapshot was found anywhere in the callee's body -- even one only
// ever referenced from loop_invariant, which --replace-call-with-contract
// never emits any check from at all, since it discards the callee's real
// body (loop included) and substitutes the contract abstraction wholesale.
// This is exactly the shape real elementwise-array-transform contracts
// need in practice (this file mirrors fm2026's array_double.c,
// increment_arr.c, reverse_array.c): a quantified old(ptr)[j] in
// loop_invariant to let function-level prove the loop inductively, with
// ensures itself stated as separate literal-indexed old(ptr[k]) conjuncts
// (github_4219_old_in_forall_array_param's ensures already covers THAT
// shape at the wrapper/enforce-mode level; this is its replace-mode call-
// site counterpart).
//
// Note on what this test does and does not exercise: the region-snapshot
// copy loop IS built here (materialize_ptr_region_old_snapshot, reused
// from the enforce-mode path) rather than aborting, but for this exact
// shape its result is provably unused -- nothing at the replaced call site
// ever reads from loop_invariant's text, so the copy loop is sliced away
// before reaching SMT encoding (confirmed: "Slicing time: ... removed N
// assignments" is measurably larger with the loop_invariant's region
// reference present than without it, for an otherwise-identical VCC
// count). What this test DOES prove is the abort is gone and the separate,
// literal-indexed ensures conjuncts genuinely constrain the caller (see
// the _fail sibling). A quantified old(ptr)[j] referenced directly inside
// ensures's own forall -- the only way to make the region-snapshot
// mechanism itself load-bearing under replace-mode -- hits a different,
// deeper issue and remains open; see
// github_4219_old_in_forall_array_param_replace_mode_ensures_knownbug.
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
