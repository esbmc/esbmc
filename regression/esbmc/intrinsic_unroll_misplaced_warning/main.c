/* Test that a misplaced __ESBMC_unroll(N) that is not directly followed by
 * a loop produces a warning instead of being silently ignored.
 */

int nondet_int();

int main()
{
  __ESBMC_unroll(5);
  int x = nondet_int();
  if(x > 0)
    x = 1;

  return x;
}
