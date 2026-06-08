/* A "lying" loop invariant that implies the loop guard, combined with a
 * non-terminating body, makes the post-loop continuation unreachable.
 * Without the vacuity probe ESBMC would silently discharge the (clearly
 * wrong) ensures clause as vacuously true. With --check-vacuity (default
 * on under --loop-invariant-check) the proof is rejected as VERIFICATION
 * UNKNOWN.
 */
extern int __VERIFIER_nondet_int(void);

int demo(int n)
{
  __ESBMC_requires(n >= 1 && n <= 10);
  __ESBMC_ensures(__ESBMC_return_value == 999);
__ESBMC_HIDE:;
  int s = 0;
  int i = 0;
  __ESBMC_loop_invariant(i <= 0);
  while (i < n)
  {
    s += 1;
    /* BUG: i not incremented -- infinite loop in reality */
  }
  return s;
}

int main()
{
  int n = __VERIFIER_nondet_int();
  __ESBMC_assume(n >= 1 && n <= 10);
  demo(n);
  return 0;
}
