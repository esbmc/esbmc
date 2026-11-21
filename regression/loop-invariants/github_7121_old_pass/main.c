/* #7121: __ESBMC_old in a loop invariant is the value on entry to the loop,
 * not the current one. */
int main()
{
  unsigned x = 5;
  unsigned i;

  __ESBMC_loop_invariant(i <= 4 && x == __ESBMC_old(x) + i);
  for (i = 0; i < 4; i++)
    x++;

  return 0;
}
