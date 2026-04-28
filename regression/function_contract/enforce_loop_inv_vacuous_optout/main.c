/* Regression test for issue #2: pin the legacy (pre-vacuity-probe)
 * behavior under --no-vacuity-check. Same lying invariant + buggy body
 * as the _unknown variant. With the probe disabled, ESBMC reverts to its
 * partial-correctness verdict of VERIFICATION SUCCESSFUL.
 *
 * This test exists so that any future change to the vacuity-probe wiring
 * which inadvertently re-enables the probe under --no-vacuity-check is
 * caught immediately.
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
