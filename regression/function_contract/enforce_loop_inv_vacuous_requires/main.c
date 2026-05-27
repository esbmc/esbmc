/* The path through this loop is unreachable for a reason other than a
 * lying invariant: the requires clause is contradictory (n >= 1 && n <= 0).
 * The vacuity probe still fires because the path assumptions are UNSAT,
 * but the verdict and warning must generalise to mention non-invariant
 * causes (over-constrained requires, upstream assume, ...) so the user
 * isn't misled into blaming the (perfectly sound) loop invariant.
 */
extern int __VERIFIER_nondet_int(void);

int demo(int n)
{
  __ESBMC_requires(n >= 1 && n <= 0);
  __ESBMC_ensures(__ESBMC_return_value == n);
__ESBMC_HIDE:;
  int i = 0;
  __ESBMC_loop_invariant(i >= 0 && i <= n);
  while (i < n)
    i++;
  return i;
}

int main()
{
  int n = __VERIFIER_nondet_int();
  demo(n);
  return 0;
}
