// Pins a deliberate design decision raised in review of #7519: the extent
// substitution in find_callsite_is_fresh_extent only rebinds names that
// match one of the callee's own formal parameters -- anything else the
// is_fresh size expression names (here, a global) is left untouched. That
// is correct, not an oversight: a global is not scoped to the callee the
// way a parameter is, so it already resolves identically at the call site
// without rewriting, the same way it resolves inside the callee's own
// is_fresh clause. This test's required pattern below fails if a future
// change makes the substitution loop try to rewrite or drop such a name
// instead of leaving it alone.
unsigned g_extent = 5 * sizeof(int);

void bump(int r[], int c)
{
  unsigned k;
  __ESBMC_requires(__ESBMC_is_fresh(r, g_extent));
  __ESBMC_ensures(r[0] == __ESBMC_old(r[0]) + c);
  __ESBMC_ensures(r[1] == __ESBMC_old(r[1]) + c);
  __ESBMC_ensures(r[2] == __ESBMC_old(r[2]) + c);
  __ESBMC_ensures(r[3] == __ESBMC_old(r[3]) + c);
  __ESBMC_ensures(r[4] == __ESBMC_old(r[4]) + c);
  __ESBMC_assigns(r[0], r[1], r[2], r[3], r[4]);

  unsigned i;
  __ESBMC_loop_invariant(
    i <= 5 &&
    __ESBMC_forall(&k, !(k < i) || (r[k] == __ESBMC_old(r)[k] + c)) &&
    __ESBMC_forall(&k, !(k >= i && k < 5) || (r[k] == __ESBMC_old(r)[k])));
  for (i = 0; i < 5; i++)
    r[i] = r[i] + c;
}

int main(void)
{
  int arr[5] = {1, 2, 3, 4, 5};
  bump(arr, 2);
  __ESBMC_assert(
    arr[0] == 3 && arr[1] == 4 && arr[2] == 5 && arr[3] == 6 && arr[4] == 7,
    "ensures propagated the correct incremented values to the caller");
  return 0;
}
