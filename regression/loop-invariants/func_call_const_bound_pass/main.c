/*
 * Regression test: loop invariant with a correct function call to a
 * constant-bound function.
 *
 * foo(a) sums a three times (constant bound i < 3), so foo(a) == 3*a.
 * The invariant foo(x) == 3*x is always true; both the base case and the
 * inductive step must pass and VERIFICATION SUCCESSFUL must be reported.
 *
 * This test guards against false negatives: the fix for vacuous passes
 * must not cause correct invariants containing function calls to fail.
 *
 * --unwind 4 is sufficient to fully evaluate foo (which iterates i < 3).
 */

unsigned foo(unsigned a)
{
  unsigned res = 0;
  for (unsigned i = 0; i < 3; ++i)
    res += a;
  return res;
}

int main()
{
  unsigned x = 0;
  __ESBMC_loop_invariant(foo(x) == 3 * x);
  for (x = 0; x < 5; ++x) {}
  return 0;
}
