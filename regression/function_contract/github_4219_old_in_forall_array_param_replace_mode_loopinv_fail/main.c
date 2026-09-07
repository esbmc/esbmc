// github_4219_old_in_forall_array_param_replace_mode_loopinv_pass with a
// caller assertion the ensures does NOT actually guarantee.
// --replace-call-with-contract assumes the callee's ensures and never
// re-examines its real body, so this -- not a broken callee implementation
// -- is what a replace-mode soundness check looks like: it proves the
// literal-indexed old() ensures conjuncts genuinely constrain the caller's
// view of the array, rather than being accepted vacuously.
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
    arr[0] == 999,
    "must fail -- the ensures guarantees arr[0] == 3, not 999");
  return 0;
}
