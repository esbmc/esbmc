/* Regression test: enforce-contract + loop-invariant (base-case violation)
 * The invariant i > 0 does not hold at loop entry (i == 0),
 * so the base-case assertion inserted by --loop-invariant must fail.
 */
int wrong_base(int n)
{
  __ESBMC_requires(n >= 1 && n <= 10);
  __ESBMC_ensures(__ESBMC_return_value == n);

  int i = 0;
  __ESBMC_loop_invariant(i > 0); /* WRONG: i == 0 at entry */
  while(i < n)
    i++;
  return i;
}

int main()
{
  wrong_base(5);
  return 0;
}
