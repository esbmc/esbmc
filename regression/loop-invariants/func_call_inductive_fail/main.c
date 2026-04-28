/*
 * Regression test: loop invariant with a function call that holds for the
 * base case but fails in the inductive step.
 *
 * foo(a) returns 3*a (constant bound i < 3).  The invariant foo(x) == 0 is
 * true initially (x=0, foo(0)=0) but false after the first loop iteration
 * (x=1, foo(1)=3 != 0).  The inductive step ASSERT must catch this and
 * VERIFICATION FAILED must be reported.
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
  __ESBMC_loop_invariant(foo(x) == 0);
  for (x = 0; x < 5; ++x) {}
  return 0;
}
