int main(void)
{
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 5);
  while (i < 5)
    i++;
  i = 100;
  __ESBMC_loop_invariant(i <= 5);
  do
  {
    i = 0;
  } while (i > 0);
  return 0;
}
