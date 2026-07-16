#include <stdbool.h>

bool eq(int a, int b)
{
  return a == b;
}

int inc(int x)
{
  return x + 1;
}

int dbl(int x)
{
  return x + x;
}

int main()
{
  int var;
  __ESBMC_assert(__ESBMC_forall(&var, eq(var + 1, 1 + var)), "forall");
  __ESBMC_assert(__ESBMC_exists(&var, eq(var + 1, 1 + var)), "exists-1");
  __ESBMC_assert(__ESBMC_exists(&var, eq(var, 6)), "exists-2");
  __ESBMC_assert(__ESBMC_exists(&var, eq(inc(var), 7)), "exists-nested");
  __ESBMC_assert(__ESBMC_exists(&var, dbl(var) == 12), "exists-dup-param");
}
