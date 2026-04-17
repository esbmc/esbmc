/* Regression test: enforce-contract + loop-invariant (ensures violation)
 * The invariant i >= 0 is too weak: after HAVOC it allows i >> n,
 * so the loop exits immediately with i != n, violating the ensures.
 */
int weak_count(int n)
{
  __ESBMC_requires(n >= 0 && n <= 10);
  __ESBMC_ensures(__ESBMC_return_value == n);

  int i = 0;
  __ESBMC_loop_invariant(i >= 0); /* too weak: cannot prove return_value == n */
  while(i < n)
    i++;
  return i;
}

int main()
{
  weak_count(5);
  return 0;
}
