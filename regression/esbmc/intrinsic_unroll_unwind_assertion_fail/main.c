/* Test that __ESBMC_unroll(N) with N smaller than the loop bound
 * triggers an unwinding assertion failure.
 * Loop runs 10 times but intrinsic only unrolls 3 times.
 */

int main()
{
  int i = 0;

  __ESBMC_unroll(3);
  while(i < 10)
    i++;

  return 0;
}
