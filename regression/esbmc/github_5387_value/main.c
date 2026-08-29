/* Companion to github_5387. The bug made the post-call path vacuous, which
 * discharges *any* assertion placed there -- so both this true assertion and
 * the false one next door reported SUCCESSFUL. Pinning both directions is what
 * distinguishes a vacuous continuation from a correctly evaluated one. */
extern int nondet_int(void);

int f(int n)
{
  if (n <= 1)
    return n;
  return f(n - 1);
}

int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n == 2 || n == 3);
  int r = f(n);
  __ESBMC_assert(r == 1, "f returns 1 for both admitted values");
  return r;
}
