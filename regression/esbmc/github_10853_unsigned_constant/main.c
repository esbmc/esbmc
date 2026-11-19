int main(void)
{
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 4 && i + 4294967295u == i - 1);
  while (i < 4)
    i++;
  return 0;
}
