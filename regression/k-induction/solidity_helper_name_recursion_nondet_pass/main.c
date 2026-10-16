int nondet_int(void);

int _ESBMC_Main_fake(int n)
{
  if (n == 0)
    return 0;
  return _ESBMC_Main_fake(n - 1);
}

int main(void)
{
  int depth = nondet_int();
  __ESBMC_assume(depth >= 1 && depth <= 3);
  __ESBMC_assert(_ESBMC_Main_fake(depth) == 0, "recursive result");
}
