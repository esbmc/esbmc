int main(void)
{
  int a[2][2][2];
  int s = 0;

  a[1][1][1] = 4;

  for (int i = 0; i < a[1][1][1]; i++)
    s++;

  __ESBMC_assert(s == 4, "a bound three dimensions deep folds");
  return 0;
}
