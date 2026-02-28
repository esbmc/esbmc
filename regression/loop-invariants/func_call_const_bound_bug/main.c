/*
 * Regression test: loop invariant containing a call to a function whose
 * internal loop has a constant upper bound.
 *
 * The C frontend extracts side-effecting sub-expressions from the invariant
 * argument into separate FUNCTION_CALL instructions placed immediately before
 * the LOOP_INVARIANT instruction.  When foo's internal loop has a constant
 * bound (i < 3), symex can kill the execution path via loop_bound_exceeded
 * before the generated ASSERT/ASSUME instructions are reached, producing 0
 * VCCs and a vacuous pass.
 *
 * The fix re-inserts fresh copies of those FUNCTION_CALL instructions
 * immediately before each ASSERT/ASSUME so that they are evaluated with the
 * correct variable state at each check point.
 *
 * --unwind 4 is sufficient to fully evaluate foo (which iterates i < 3).
 * The invariant foo(x)+1==foo(x) is always false; the base-case ASSERT must
 * fail and VERIFICATION FAILED must be reported.
 */

unsigned short foo(unsigned short a)
{
  unsigned short res = 0;
  for (unsigned short i = 0; i < 3; ++i)
    res += a;
  return res;
}

int main()
{
  unsigned short idx;
  unsigned short res = 0;
  __ESBMC_loop_invariant(foo(idx) + 1 == foo(idx));
  for (idx = 0; idx < 7; ++idx)
    res += idx;
  return 0;
}
