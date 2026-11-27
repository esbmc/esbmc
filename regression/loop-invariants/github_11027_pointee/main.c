/* Regression: GitHub #11027 -- the loop writes a[] only through p, so its
 * havoc is of the pointee. a[0] is 0 on the first iteration, so the run from
 * the loop's entry state must keep a's entry value: left free it reports a
 * false failure. */
int main()
{
  int a[4] = {0, 0, 0, 0};
  int *p = a;
  int i;
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  for (i = 0; i < 4; ++i)
  {
    __ESBMC_assert(a[0] == 0 || i >= 1, "a0 is zero on the first iteration");
    p[i] = 1;
  }
}
