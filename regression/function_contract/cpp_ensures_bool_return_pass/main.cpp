/* cpp_ensures_bool_return_pass:
 * --enforce-contract on a bool-returning function whose ensures compares
 * __ESBMC_return_value against a boolean-typed expression.
 *
 * The global __ESBMC_return_value is declared `int`, so Clang promotes the
 * boolean operand of the equality to `int`. Once the symbol's type is
 * corrected to the function's real `bool` return type, the comparison was left
 * with one side Bool and the other a (_ BitVec 32), which crashed Z3 with
 * "Sorts Bool and (_ BitVec 32) are incompatible" and a core dump.
 *
 * The body returns exactly that boolean, so the postcondition holds.
 *
 * Regression for: https://github.com/Yiannis128/esbmc/issues/4
 *
 * Expected: VERIFICATION SUCCESSFUL
 */

bool is_pos(int c)
{
  __ESBMC_ensures(__ESBMC_return_value == (c > 0));
  return c > 0;
}
