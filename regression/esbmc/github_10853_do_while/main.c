int main(void)
{
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 10);
  do
  {
    i++;
  } while (i < 10);
  return 0;
}
