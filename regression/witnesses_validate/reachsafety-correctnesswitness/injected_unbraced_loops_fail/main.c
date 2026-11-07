int main(void)
{
  int x = 0;
  unsigned int i, j;
  for (i = 0; i < 2; ++i)
    for (j = 0; j < 2; ++j)
      ++x;
  __ESBMC_assert(x == 2, "iteration count");
  return 0;
}
