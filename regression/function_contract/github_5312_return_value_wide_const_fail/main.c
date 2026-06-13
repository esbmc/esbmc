/* Issue #5312 negative variant: the same wide-constant comparison path, but the
 * ensures is genuinely violated. The width fix must still encode the comparison
 * (no sort-mismatch abort) and the verdict must be FAILED: f returns a small
 * positive value, which is never greater than INT_MAX + 1.
 */

int f(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_requires(x < 16);
  __ESBMC_ensures(__ESBMC_return_value > 2147483648); /* INT_MAX + 1 */
  return x;
}

int main()
{
  int r = f(5);
  return r;
}
