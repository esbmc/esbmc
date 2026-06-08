/* Test __ESBMC_unroll(N) with a for loop that declares and updates
 * several induction variables. The preamble between the intrinsic and the
 * loop head holds multiple DECL/ASSIGN instructions, all of which must be
 * skipped when binding the intrinsic to the loop.
 * The loop runs 5 times, so unroll(10) fully covers it.
 */

int main()
{
  int sum = 0;

  __ESBMC_unroll(10);
  for(int i = 0, j = 10; i < j; i++, j--)
    sum += i;

  return 0;
}
