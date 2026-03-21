/* Test __ESBMC_unroll(N) with a while loop.
 * Loop bound is 10, intrinsic sets unroll to 10.
 * Verification succeeds with no unwinding assertions.
 */

int main()
{
  int i = 0;

  __ESBMC_unroll(10);
  while(i < 10)
    i++;

  return 0;
}
