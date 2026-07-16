_Bool eq(int a, int b)
{
  return a == b;
}

int main()
{
  int var;
  __ESBMC_assert(__ESBMC_forall(&var, eq(var + 1, 1 + var)), "forall");
  __ESBMC_assert(__ESBMC_exists(&var, eq(var + 1, 1 + var)), "exists-1");
  __ESBMC_assert(__ESBMC_exists(&var, eq(var, 6)), "exists-2");
}
