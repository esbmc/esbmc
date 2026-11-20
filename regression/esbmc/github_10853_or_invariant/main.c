int main(void)
{
  unsigned i = 0, j = 0;
  __ESBMC_loop_invariant(i <= 10);
  while (i < 10)
    i++;
  i = 20;
  __ESBMC_loop_invariant(j <= 5 || i <= 10);
  while (j < 5)
    j++;
  return 0;
}
