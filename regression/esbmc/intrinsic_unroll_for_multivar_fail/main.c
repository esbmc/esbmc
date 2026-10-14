/* Test __ESBMC_unroll(N) with N smaller than the iteration count of a
 * multi-variable for loop. The loop runs 5 times but the intrinsic only
 * unrolls 2 times, so an unwinding assertion must fail.
 */

int main()
{
  int sum = 0;

  __ESBMC_unroll(2);
  for(int i = 0, j = 10; i < j; i++, j--)
    sum += i;

  return 0;
}
