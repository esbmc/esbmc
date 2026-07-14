/* Issue #5312: --enforce-contract aborted with a Z3/Bitwuzla sort mismatch
 * (BitVec 64 vs BitVec 32) when an __ESBMC_ensures compared the 32-bit
 * __ESBMC_return_value against a constant that does not fit in int.
 *
 * In C such a literal has type long (64-bit), so the comparison is encoded at
 * 64 bits; the contract machinery stripped the usual-arithmetic-conversion
 * cast off the return value, leaving a 32-bit operand against the 64-bit
 * constant. The ensures is genuinely true: an int return value is always
 * greater than INT_MIN - 1 and always less than INT_MAX + 1.
 */

int f(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_requires(x < 16);
  __ESBMC_ensures(__ESBMC_return_value > -2147483649); /* INT_MIN - 1 */
  __ESBMC_ensures(__ESBMC_return_value < 2147483648);  /* INT_MAX + 1 */
  return x;
}

int main()
{
  int r = f(5);
  return r;
}
