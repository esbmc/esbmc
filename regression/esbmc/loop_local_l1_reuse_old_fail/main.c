int main()
{
  int *q = 0;
  for (int i = 0; i < 2; i++)
  {
    int x = i;
    if (i == 0)
      q = &__ESBMC_old(x);
    else
      return *q;
  }
  return 0;
}
