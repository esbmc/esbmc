/* cpp_ensures_bool_return_fail:
 * Negative variant of cpp_ensures_bool_return_pass. Same bool-returning
 * contract shape (which previously crashed Z3 with a Bool vs (_ BitVec 32)
 * sort error), but the body returns `c >= 0`, which disagrees with the
 * postcondition `__ESBMC_return_value == (c > 0)` when c == 0.
 *
 * This confirms the sort fix does not mask a genuine contract violation.
 *
 * Regression for: https://github.com/Yiannis128/esbmc/issues/4
 *
 * Expected: VERIFICATION FAILED
 */

bool is_pos(int c)
{
  __ESBMC_ensures(__ESBMC_return_value == (c > 0));
  return c >= 0; /* VIOLATION: differs from (c > 0) at c == 0 */
}
