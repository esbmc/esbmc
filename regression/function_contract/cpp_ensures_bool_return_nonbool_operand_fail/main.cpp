/* cpp_ensures_bool_return_nonbool_operand_fail:
 * Guards the *direction* of the bool/bitvector fix for issue #4.
 *
 * The ensures compares the bool return value against `x & 3`, which can be 2
 * or 3 — values outside {0,1}. The body returns `x & 3`, but the bool return
 * type collapses it to {0,1}, so the postcondition is violated whenever
 * x & 3 >= 2.
 *
 * The sort fix must *promote* the boolean operand to int (C usual arithmetic
 * conversions), not demote `x & 3` to bool. Demoting would collapse the
 * operand to {0,1} and wrongly report VERIFICATION SUCCESSFUL, masking this
 * real violation.
 *
 * Regression for: https://github.com/Yiannis128/esbmc/issues/4
 *
 * Expected: VERIFICATION FAILED
 */

bool f(int x)
{
  __ESBMC_ensures(__ESBMC_return_value == (x & 3));
  return x & 3; /* VIOLATION: bool return collapses 2,3 to 1 */
}
