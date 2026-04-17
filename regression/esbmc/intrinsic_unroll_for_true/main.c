/* Test __ESBMC_unroll(N) with a for loop.
 * Loop bound is 5, intrinsic sets unroll to 5.
 * Verification succeeds with no unwinding assertions.
 */

int main()
{
  int sum = 0;

  __ESBMC_unroll(5);
  for(int i = 0; i < 5; i++)
    sum += i;

  return 0;
}
