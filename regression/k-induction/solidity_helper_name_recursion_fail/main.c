int _ESBMC_Main_fake(int n)
{
  if (n == 0)
    return 0;
  return _ESBMC_Main_fake(n - 1);
}

int main(void)
{
  __ESBMC_assert(_ESBMC_Main_fake(3) != 0, "recursive result");
}
