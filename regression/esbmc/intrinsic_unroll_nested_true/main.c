/* Test that __ESBMC_unroll(N) binds to the inner of two nested loops and
 * never to the enclosing loop. The intrinsic sits inside the outer loop,
 * immediately before the inner loop. Only the inner loop must be annotated.
 * The outer loop is bounded so the program still verifies.
 */

int main()
{
  int sum = 0;

  for(int k = 0; k < 3; k++)
  {
    __ESBMC_unroll(10);
    for(int i = 0, j = 10; i < j; i++, j--)
      sum += i;
  }

  return 0;
}
