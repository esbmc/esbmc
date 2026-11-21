/* #7121: the loop changes x, so x == __ESBMC_old(x) is not inductive. It held
 * trivially while __ESBMC_old read the current value. */
int main()
{
  unsigned x = 5;
  unsigned i;

  __ESBMC_loop_invariant(i <= 4 && x == __ESBMC_old(x));
  for (i = 0; i < 4; i++)
    x++;

  return 0;
}
